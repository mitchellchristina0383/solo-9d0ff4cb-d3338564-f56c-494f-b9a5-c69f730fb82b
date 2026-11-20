"""Job model and on-disk job store.

Every job owns a directory ``<output_root>/<job_id>/``::

    job.json                 job state, rewritten after every finished shard
    rank_00000/weights.f32   resharded weights for rank 0
    rank_00000/momentum.f32  resharded momentum for rank 0
    rank_00000/shard.json    {"rank", "start", "end", "step"} — written LAST,
                             its presence marks the shard as complete

The job id is a deterministic hash of (source dir, manifest content, new
world size), so re-issuing the same reshard request returns the same job and
simply resumes it instead of starting over.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import FLOAT32_BYTES, Manifest

RANK_DIR_FMT = "rank_{:05d}"
WEIGHTS_NAME = "weights.f32"
MOMENTUM_NAME = "momentum.f32"
SHARD_META_NAME = "shard.json"
JOB_META_NAME = "job.json"

TERMINAL_STATES = {"done", "failed"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compute_job_id(source_dir: Path, manifest: Manifest, world_size: int) -> str:
    canonical = json.dumps(
        {
            "source_dir": str(source_dir.resolve()),
            "step": manifest.step,
            "total_length": manifest.total_length,
            "weights": [(s.path, s.start, s.end) for s in manifest.weights],
            "momentum": [(s.path, s.start, s.end) for s in manifest.momentum],
            "world_size": world_size,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class RankState:
    rank: int
    start: int
    end: int
    status: str = "pending"  # pending | done

    def to_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "start": self.start, "end": self.end, "status": self.status}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RankState":
        return cls(rank=int(data["rank"]), start=int(data["start"]), end=int(data["end"]),
                   status=str(data.get("status", "pending")))


@dataclass
class Job:
    job_id: str
    source_dir: str
    world_size: int
    step: int
    total_length: int
    ranks: list[RankState]
    status: str = "pending"  # pending | running | done | failed
    error: str | None = None
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "source_dir": self.source_dir,
            "world_size": self.world_size,
            "step": self.step,
            "total_length": self.total_length,
            "status": self.status,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "ranks": [r.to_dict() for r in self.ranks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        return cls(
            job_id=str(data["job_id"]),
            source_dir=str(data["source_dir"]),
            world_size=int(data["world_size"]),
            step=int(data["step"]),
            total_length=int(data["total_length"]),
            status=str(data.get("status", "pending")),
            error=data.get("error"),
            created_at=str(data.get("created_at", _utcnow())),
            updated_at=str(data.get("updated_at", _utcnow())),
            ranks=[RankState.from_dict(r) for r in data.get("ranks", [])],
        )

    @property
    def done_ranks(self) -> int:
        return sum(1 for r in self.ranks if r.status == "done")


class JobStore:
    """Loads and persists jobs under the output root.  Thread-safe."""

    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._load_all()

    def _load_all(self) -> None:
        for entry in sorted(self.output_root.iterdir()):
            meta = entry / JOB_META_NAME
            if not entry.is_dir() or not meta.is_file():
                continue
            try:
                job = Job.from_dict(json.loads(meta.read_text(encoding="utf-8")))
            except (OSError, ValueError, KeyError):
                continue  # corrupt job dir: leave it alone, don't crash startup
            # A process restart interrupts whatever was running; the shards on
            # disk are the source of truth, so interrupted jobs are resumable.
            if job.status == "running":
                job.status = "failed"
                job.error = "interrupted by service restart; call resume to continue"
            self._jobs[job.job_id] = job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at)

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.job_id] = job
        self.save(job)

    def save(self, job: Job) -> None:
        job.updated_at = _utcnow()
        job_dir = self.job_dir(job.job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(job.to_dict(), indent=2) + "\n"
        # Atomic replace so a crash never leaves a half-written job.json.
        fd, tmp = tempfile.mkstemp(dir=job_dir, prefix=".job-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, job_dir / JOB_META_NAME)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def job_dir(self, job_id: str) -> Path:
        return self.output_root / job_id

    def rank_dir(self, job_id: str, rank: int) -> Path:
        return self.job_dir(job_id) / RANK_DIR_FMT.format(rank)


def shard_is_complete(rank_dir: Path, expected_elements: int) -> bool:
    """A shard counts as complete only if its marker (shard.json, written
    last) exists and both data files have the exact expected byte size."""
    meta = rank_dir / SHARD_META_NAME
    if not meta.is_file():
        return False
    expected_bytes = expected_elements * FLOAT32_BYTES
    for name in (WEIGHTS_NAME, MOMENTUM_NAME):
        f = rank_dir / name
        if not f.is_file() or f.stat().st_size != expected_bytes:
            return False
    return True
