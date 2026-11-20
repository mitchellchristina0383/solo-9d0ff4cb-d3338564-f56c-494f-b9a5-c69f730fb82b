"""FastAPI surface for the checkpoint resharding service.

Flow (all interactive via /docs):
  POST /reshard                              plan & start a conversion
  GET  /jobs/{job_id}                        poll progress (intervals + step)
  POST /jobs/{job_id}/resume                 continue after failure/restart
  GET  /jobs/{job_id}/shards/{rank}/meta     interval + original step per rank
  GET  /jobs/{job_id}/shards/{rank}/download?part=weights|momentum
                                             gated: 409 until the whole job is done
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .engine import DEFAULT_CHUNK_BYTES, execute_job
from .jobs import (
    MOMENTUM_NAME,
    WEIGHTS_NAME,
    Job,
    JobStore,
    RankState,
    compute_job_id,
)
from .manifest import ManifestError, load_manifest
from .plans import rank_intervals


class Settings(BaseModel):
    source_dir: Path = Path(os.environ.get("SOURCE_DIR", "/data/source"))
    output_root: Path = Path(os.environ.get("OUTPUT_ROOT", "/data/outputs"))
    chunk_bytes: int = int(os.environ.get("CHUNK_BYTES", DEFAULT_CHUNK_BYTES))
    sync_execution: bool = os.environ.get("RESHARD_SYNC", "").lower() in {"1", "true", "yes"}


class ReshardRequest(BaseModel):
    new_world_size: int = Field(..., ge=1, description="number of processes to reshard for")
    source_dir: str | None = Field(
        None, description="override the mounted source directory (defaults to SOURCE_DIR)"
    )


class RankOut(BaseModel):
    rank: int
    start: int
    end: int
    status: str


class JobOut(BaseModel):
    job_id: str
    status: str
    error: str | None
    step: int
    total_length: int
    world_size: int
    done_ranks: int
    created_at: str
    updated_at: str
    ranks: list[RankOut]


def _job_out(job: Job) -> JobOut:
    return JobOut(
        job_id=job.job_id,
        status=job.status,
        error=job.error,
        step=job.step,
        total_length=job.total_length,
        world_size=job.world_size,
        done_ranks=job.done_ranks,
        created_at=job.created_at,
        updated_at=job.updated_at,
        ranks=[RankOut(rank=r.rank, start=r.start, end=r.end, status=r.status) for r in job.ranks],
    )


class Runner:
    """Serialises execution per job; threads for async, inline when sync_execution."""

    def __init__(self, store: JobStore, chunk_bytes: int, sync: bool):
        self._store = store
        self._chunk = chunk_bytes
        self._sync = sync
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, job_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(job_id, threading.Lock())

    def kick(self, job: Job) -> None:
        if job.status == "done":
            return  # nothing left to do
        lock = self._lock_for(job.job_id)
        if job.status == "running" and lock.locked():
            return  # already running elsewhere

        def _run() -> None:
            with lock:
                execute_job(self._store, job, self._chunk)

        if self._sync:
            _run()
        else:
            threading.Thread(target=_run, daemon=True, name=f"reshard-{job.job_id}").start()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    store = JobStore(settings.output_root)
    runner = Runner(store, settings.chunk_bytes, settings.sync_execution)

    app = FastAPI(
        title="Checkpoint Reshard API",
        version="1.0.0",
        description=(
            "Re-shard a float32 training checkpoint (weights + momentum) from a "
            "read-only source volume to a new process count. Source segments with "
            "gaps or overlaps are rejected. Fully written rank shards are reused "
            "on resume; downloads open only when the whole job is complete."
        ),
    )
    app.state.store = store
    app.state.runner = runner
    app.state.settings = settings

    def get_store() -> JobStore:
        return store

    def must_job(job_id: str) -> Job:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
        return job

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/reshard", response_model=JobOut, status_code=202)
    def create_reshard(req: ReshardRequest) -> JobOut:
        source_dir = Path(req.source_dir) if req.source_dir else settings.source_dir
        if not source_dir.is_dir():
            raise HTTPException(status_code=422, detail=f"source directory {source_dir} not found")
        try:
            manifest = load_manifest(source_dir)
        except ManifestError as exc:
            # gaps / overlaps / size mismatches all land here
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            intervals = rank_intervals(manifest.total_length, req.new_world_size)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        job_id = compute_job_id(source_dir, manifest, req.new_world_size)
        job = store.get(job_id)
        if job is None:
            job = Job(
                job_id=job_id,
                source_dir=str(source_dir),
                world_size=req.new_world_size,
                step=manifest.step,
                total_length=manifest.total_length,
                ranks=[RankState(rank=i, start=s, end=e) for i, (s, e) in enumerate(intervals)],
            )
            store.add(job)
        runner.kick(job)
        return _job_out(job)

    @app.get("/jobs", response_model=list[JobOut])
    def list_jobs() -> list[JobOut]:
        return [_job_out(j) for j in store.all()]

    @app.get("/jobs/{job_id}", response_model=JobOut)
    def get_job(job: Job = Depends(must_job)) -> JobOut:
        return _job_out(job)

    @app.post("/jobs/{job_id}/resume", response_model=JobOut, status_code=202)
    def resume_job(job: Job = Depends(must_job)) -> JobOut:
        if job.status == "done":
            return _job_out(job)  # nothing to do
        runner.kick(job)
        return _job_out(job)

    @app.get("/jobs/{job_id}/shards/{rank}/meta")
    def shard_meta(rank: int, job: Job = Depends(must_job)) -> dict:
        if not 0 <= rank < job.world_size:
            raise HTTPException(status_code=404, detail=f"rank {rank} out of range")
        r = job.ranks[rank]
        return {
            "job_id": job.job_id,
            "rank": r.rank,
            "start": r.start,
            "end": r.end,
            "step": job.step,
            "status": r.status,
        }

    @app.get("/jobs/{job_id}/shards/{rank}/download")
    def download_shard(
        rank: int,
        part: Literal["weights", "momentum"] = Query(...),
        job: Job = Depends(must_job),
    ) -> FileResponse:
        if not 0 <= rank < job.world_size:
            raise HTTPException(status_code=404, detail=f"rank {rank} out of range")
        if job.status != "done":
            raise HTTPException(
                status_code=409,
                detail=(
                    f"job is {job.status}: downloads open only when all "
                    f"{job.world_size} shards are complete ({job.done_ranks} done)"
                ),
            )
        name = WEIGHTS_NAME if part == "weights" else MOMENTUM_NAME
        path = store.rank_dir(job.job_id, rank) / name
        if not path.is_file():
            raise HTTPException(status_code=409, detail=f"shard {rank} {part} not on disk yet")
        return FileResponse(path, media_type="application/octet-stream",
                            filename=f"{job.job_id}-rank{rank:05d}-{name}")

    return app


# Run with: uvicorn app.main:create_app --factory
# (no module-level app: importing this module must not touch the filesystem)
