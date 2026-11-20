import json

import pytest

from app.engine import execute_job
from app.jobs import Job, JobStore, RankState, compute_job_id, shard_is_complete
from app.manifest import load_manifest
from app.plans import rank_intervals
from tests.conftest import write_source

TOTAL = 10007
STEP = 7459
WORLD = 5


def make_source(tmp_path):
    globals_ = write_source(
        tmp_path / "src",
        total=TOTAL,
        step=STEP,
        weight_bounds=[0, 3500, 7200, TOTAL],
        momentum_bounds=[0, 2000, 9000, TOTAL],
    )
    return tmp_path / "src", globals_


def make_job(store: JobStore, source_dir, world=WORLD) -> Job:
    manifest = load_manifest(source_dir)
    job_id = compute_job_id(source_dir, manifest, world)
    job = Job(
        job_id=job_id,
        source_dir=str(source_dir),
        world_size=world,
        step=manifest.step,
        total_length=manifest.total_length,
        ranks=[
            RankState(rank=i, start=s, end=e)
            for i, (s, e) in enumerate(rank_intervals(manifest.total_length, world))
        ],
    )
    store.add(job)
    return job


def assert_rank_bytes(store, job, globals_):
    for r in job.ranks:
        rank_dir = store.rank_dir(job.job_id, r.rank)
        w = (rank_dir / "weights.f32").read_bytes()
        m = (rank_dir / "momentum.f32").read_bytes()
        assert w == globals_["weights"][r.start * 4 : r.end * 4]
        assert m == globals_["momentum"][r.start * 4 : r.end * 4]
        meta = json.loads((rank_dir / "shard.json").read_text())
        assert meta == {
            "rank": r.rank,
            "start": r.start,
            "end": r.end,
            "step": STEP,
            "dtype": "float32",
            "files": ["weights.f32", "momentum.f32"],
        }


class TestEndToEnd:
    def test_full_reshard_byte_exact(self, tmp_path):
        source, globals_ = make_source(tmp_path)
        store = JobStore(tmp_path / "out")
        job = execute_job(store, make_job(store, source), chunk_bytes=1024)  # tiny chunks
        assert job.status == "done", job.error
        assert job.done_ranks == WORLD
        assert_rank_bytes(store, job, globals_)

    def test_different_split_points_weights_vs_momentum(self, tmp_path):
        # weights 3 parts, momentum 3 different parts -> rank files must still
        # be exact slices of each global tensor (covered by byte equality above)
        source, globals_ = make_source(tmp_path)
        store = JobStore(tmp_path / "out")
        job = execute_job(store, make_job(store, source, world=4))
        assert job.status == "done"
        assert_rank_bytes(store, job, globals_)


class TestResume:
    def test_incomplete_shards_redone_complete_ones_reused(self, tmp_path):
        source, globals_ = make_source(tmp_path)
        store = JobStore(tmp_path / "out")
        job = execute_job(store, make_job(store, source))
        assert job.status == "done"

        # Snapshot mtimes of complete shards.
        untouched = {}
        for rank in (0, 1, 3, 4):
            f = store.rank_dir(job.job_id, rank) / "weights.f32"
            untouched[rank] = f.stat().st_mtime_ns

        # Simulate an interrupted rank 2: marker gone, weights truncated,
        # stale tmp file left behind.
        rank2 = store.rank_dir(job.job_id, 2)
        (rank2 / "shard.json").unlink()
        (rank2 / "weights.f32").write_bytes(b"\x00" * 16)
        (rank2 / "momentum.f32.tmp").write_bytes(b"\x00" * 8)
        job.ranks[2].status = "pending"
        job.status = "failed"
        store.save(job)

        job = execute_job(store, job)
        assert job.status == "done", job.error
        assert_rank_bytes(store, job, globals_)
        for rank, mtime in untouched.items():
            f = store.rank_dir(job.job_id, rank) / "weights.f32"
            assert f.stat().st_mtime_ns == mtime, f"rank {rank} was rewritten"
        assert not (rank2 / "momentum.f32.tmp").exists()

    def test_restart_with_fresh_store_reuses_disk_state(self, tmp_path):
        source, globals_ = make_source(tmp_path)
        store = JobStore(tmp_path / "out")
        job = execute_job(store, make_job(store, source))
        assert job.status == "done"

        # Corrupt one shard, then simulate a service restart: brand-new
        # JobStore reading the same output root.
        rank1 = store.rank_dir(job.job_id, 1)
        (rank1 / "shard.json").unlink()
        (rank1 / "weights.f32").unlink()

        store2 = JobStore(tmp_path / "out")
        job2 = store2.get(job.job_id)
        assert job2 is not None
        job2 = execute_job(store2, job2)
        assert job2.status == "done", job2.error
        assert_rank_bytes(store2, job2, globals_)

    def test_running_job_marked_failed_on_restart(self, tmp_path):
        source, _ = make_source(tmp_path)
        store = JobStore(tmp_path / "out")
        job = make_job(store, source)
        job.status = "running"
        store.save(job)

        store2 = JobStore(tmp_path / "out")
        reloaded = store2.get(job.job_id)
        assert reloaded.status == "failed"
        assert "restart" in reloaded.error

    def test_source_change_detected_on_resume(self, tmp_path):
        source, _ = make_source(tmp_path)
        store = JobStore(tmp_path / "out")
        job = execute_job(store, make_job(store, source))
        assert job.status == "done"

        # Same layout, different step -> the checkpoint moved on.
        manifest = json.loads((source / "manifest.json").read_text())
        manifest["step"] = STEP + 1
        (source / "manifest.json").write_text(json.dumps(manifest))

        job.status = "failed"
        job = execute_job(store, job)
        assert job.status == "failed"
        assert "changed" in job.error


class TestShardCompleteness:
    def test_marker_and_sizes_required(self, tmp_path):
        rank_dir = tmp_path / "rank_00000"
        rank_dir.mkdir()
        assert not shard_is_complete(rank_dir, 10)
        (rank_dir / "weights.f32").write_bytes(b"\x00" * 40)
        (rank_dir / "momentum.f32").write_bytes(b"\x00" * 40)
        assert not shard_is_complete(rank_dir, 10)  # no marker yet
        (rank_dir / "shard.json").write_text("{}")
        assert shard_is_complete(rank_dir, 10)
        (rank_dir / "weights.f32").write_bytes(b"\x00" * 39)
        assert not shard_is_complete(rank_dir, 10)
