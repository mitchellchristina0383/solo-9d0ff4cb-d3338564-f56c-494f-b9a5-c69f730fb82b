import json

import pytest
from fastapi.testclient import TestClient

from app.main import Settings, create_app
from tests.conftest import write_source

TOTAL = 1000
STEP = 321


@pytest.fixture()
def client(tmp_path):
    source = tmp_path / "src"
    globals_ = write_source(
        source,
        total=TOTAL,
        step=STEP,
        weight_bounds=[0, 400, TOTAL],
        momentum_bounds=[0, 250, 800, TOTAL],
    )
    app = create_app(Settings(source_dir=source, output_root=tmp_path / "out",
                              chunk_bytes=256, sync_execution=True))
    with TestClient(app) as c:
        c.globals_ = globals_
        c.source = source
        yield c


def create_job(client, world=4):
    resp = client.post("/reshard", json={"new_world_size": world})
    assert resp.status_code == 202, resp.text
    return resp.json()


class TestReshardFlow:
    def test_create_poll_download(self, client):
        job = create_job(client, world=4)
        assert job["status"] == "done"  # sync execution in tests
        assert job["step"] == STEP
        assert job["total_length"] == TOTAL
        assert len(job["ranks"]) == 4

        # Poll status.
        again = client.get(f"/jobs/{job['job_id']}").json()
        assert again["status"] == "done"
        assert again["done_ranks"] == 4

        # Intervals are contiguous, near-even, and cover the tensor.
        ranks = again["ranks"]
        assert ranks[0]["start"] == 0 and ranks[-1]["end"] == TOTAL
        for a, b in zip(ranks, ranks[1:]):
            assert a["end"] == b["start"]

        # Per-rank meta exposes interval + original step.
        meta = client.get(f"/jobs/{job['job_id']}/shards/2/meta").json()
        assert meta["step"] == STEP
        assert meta["start"] == ranks[2]["start"] and meta["end"] == ranks[2]["end"]

        # Downloads are byte-exact slices of the global tensors.
        for r in ranks:
            for part in ("weights", "momentum"):
                resp = client.get(
                    f"/jobs/{job['job_id']}/shards/{r['rank']}/download",
                    params={"part": part},
                )
                assert resp.status_code == 200
                expected = client.globals_[part][r["start"] * 4 : r["end"] * 4]
                assert resp.content == expected

    def test_same_request_returns_same_job(self, client):
        first = create_job(client, world=3)
        second = create_job(client, world=3)
        assert first["job_id"] == second["job_id"]

    def test_list_jobs(self, client):
        create_job(client, world=2)
        create_job(client, world=5)
        jobs = client.get("/jobs").json()
        assert len(jobs) == 2


class TestDownloadGate:
    def test_download_blocked_until_whole_job_done(self, client):
        job = create_job(client, world=4)
        # Break one shard on disk and force the job back to an unfinished state.
        store = client.app.state.store
        rank1 = store.rank_dir(job["job_id"], 1)
        (rank1 / "shard.json").unlink()
        stored = store.get(job["job_id"])
        stored.status = "failed"
        stored.ranks[1].status = "pending"
        store.save(stored)

        resp = client.get(f"/jobs/{job['job_id']}/shards/0/download",
                          params={"part": "weights"})
        assert resp.status_code == 409
        assert "downloads open only" in resp.json()["detail"]

        # Resume finishes the missing shard; downloads open up.
        resumed = client.post(f"/jobs/{job['job_id']}/resume")
        assert resumed.status_code == 202
        assert resumed.json()["status"] == "done"
        resp = client.get(f"/jobs/{job['job_id']}/shards/0/download",
                          params={"part": "weights"})
        assert resp.status_code == 200
        r0 = client.get(f"/jobs/{job['job_id']}").json()["ranks"][0]
        assert resp.content == client.globals_["weights"][r0["start"] * 4 : r0["end"] * 4]


class TestValidation:
    def test_gap_rejected_422(self, client):
        manifest = json.loads((client.source / "manifest.json").read_text())
        manifest["weights"][1]["start"] += 1  # open a gap
        (client.source / "manifest.json").write_text(json.dumps(manifest))
        resp = client.post("/reshard", json={"new_world_size": 4})
        assert resp.status_code == 422
        assert "gap" in resp.json()["detail"]

    def test_overlap_rejected_422(self, client):
        manifest = json.loads((client.source / "manifest.json").read_text())
        manifest["momentum"][1]["start"] -= 1  # overlap with previous segment
        (client.source / "manifest.json").write_text(json.dumps(manifest))
        resp = client.post("/reshard", json={"new_world_size": 4})
        assert resp.status_code == 422
        assert "overlap" in resp.json()["detail"]

    def test_world_size_too_large_rejected(self, client):
        resp = client.post("/reshard", json={"new_world_size": TOTAL + 1})
        assert resp.status_code == 422

    def test_missing_source_dir_422(self, client):
        resp = client.post("/reshard", json={"new_world_size": 2,
                                             "source_dir": "/nonexistent"})
        assert resp.status_code == 422

    def test_unknown_job_404(self, client):
        assert client.get("/jobs/nope").status_code == 404
        assert client.post("/jobs/nope/resume").status_code == 404
