"""Resharding engine: chunked, crash-safe materialisation of rank shards."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .jobs import (
    MOMENTUM_NAME,
    SHARD_META_NAME,
    WEIGHTS_NAME,
    Job,
    JobStore,
    shard_is_complete,
)
from .manifest import FLOAT32_BYTES, Manifest, load_manifest
from .plans import CopyOp, plan_copy_ops, rank_intervals

DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024  # 4 MiB


class SourceChangedError(RuntimeError):
    """The source checkpoint no longer matches the one the job was planned from."""


def _copy_ops_to_file(source_dir: Path, ops: list[CopyOp], dst: Path, chunk_bytes: int) -> None:
    """Stream the given source ranges into ``dst`` via a temp file + atomic rename."""
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        tmp.unlink()  # leftover from an interrupted attempt
    try:
        with open(tmp, "wb") as out:
            for op in ops:
                src = source_dir / op.path
                remaining = op.length * FLOAT32_BYTES
                offset = op.src_offset * FLOAT32_BYTES
                with open(src, "rb", buffering=0) as fh:
                    while remaining > 0:
                        # pread with an explicit offset keeps every read
                        # positioned regardless of file-object state.
                        data = os.pread(fh.fileno(), min(chunk_bytes, remaining), offset)
                        if not data:
                            raise IOError(
                                f"short read on {op.path} at byte {offset}: "
                                f"source changed underneath us?"
                            )
                        out.write(data)
                        offset += len(data)
                        remaining -= len(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def _write_shard_meta(rank_dir: Path, rank: int, start: int, end: int, step: int) -> None:
    """Write shard.json LAST — it is the completeness marker for the shard."""
    payload = json.dumps(
        {"rank": rank, "start": start, "end": end, "step": step,
         "dtype": "float32", "files": [WEIGHTS_NAME, MOMENTUM_NAME]},
        indent=2,
    ) + "\n"
    fd, tmp = tempfile.mkstemp(dir=rank_dir, prefix=".shard-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, rank_dir / SHARD_META_NAME)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _verify_source_unchanged(job: Job, manifest: Manifest) -> None:
    if manifest.step != job.step or manifest.total_length != job.total_length:
        raise SourceChangedError(
            f"source checkpoint changed: job was planned for step={job.step} "
            f"total_length={job.total_length}, source now has step={manifest.step} "
            f"total_length={manifest.total_length}"
        )


def execute_job(store: JobStore, job: Job, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> Job:
    """Run/resume a job synchronously.

    Shards that are already fully on disk (marker + exact sizes) are reused
    untouched; only incomplete shards are (re)written.  Progress is persisted
    after every shard, so a crash leaves a resumable state.
    """
    job.status = "running"
    job.error = None
    store.save(job)
    try:
        source_dir = Path(job.source_dir)
        manifest = load_manifest(source_dir)  # re-validates gaps/overlaps/sizes
        _verify_source_unchanged(job, manifest)

        intervals = rank_intervals(job.total_length, job.world_size)
        for rank_state in job.ranks:
            start, end = intervals[rank_state.rank]
            if (start, end) != (rank_state.start, rank_state.end):
                raise SourceChangedError(
                    f"rank {rank_state.rank} interval changed: "
                    f"stored [{rank_state.start}, {rank_state.end}), now [{start}, {end})"
                )
            rank_dir = store.rank_dir(job.job_id, rank_state.rank)
            if shard_is_complete(rank_dir, end - start):
                # Reuse the fully written shard — also covers a crash between
                # writing shard.json and persisting job state.
                rank_state.status = "done"
                store.save(job)
                continue
            rank_dir.mkdir(parents=True, exist_ok=True)
            for segments, name in (
                (manifest.weights, WEIGHTS_NAME),
                (manifest.momentum, MOMENTUM_NAME),
            ):
                ops = plan_copy_ops(segments, start, end)
                _copy_ops_to_file(source_dir, ops, rank_dir / name, chunk_bytes)
            _write_shard_meta(rank_dir, rank_state.rank, start, end, job.step)
            rank_state.status = "done"
            store.save(job)  # persist progress after each shard

        job.status = "done"
    except Exception as exc:  # noqa: BLE001 — any failure must be persisted, not lost
        job.status = "failed"
        job.error = str(exc)
    store.save(job)
    return job
