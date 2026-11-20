"""Training-step reconciliation: runs, scheduler reports, kept/lost steps."""

import threading

import pytest
from fastapi.testclient import TestClient

from app.main import Settings, create_app
from app.runs import RunStore
from tests.conftest import write_source

TOTAL = 1000


@pytest.fixture()
def client(tmp_path):
    app = create_app(
        Settings(
            source_dir=tmp_path / "no-default-source",
            output_root=tmp_path / "out",
            chunk_bytes=256,
            sync_execution=True,
        )
    )
    with TestClient(app) as c:
        c.tmp_path = tmp_path
        yield c


def make_done_job(client, step, world=2):
    """Create a source checkpoint at ``step`` and reshard it to completion."""
    src = client.tmp_path / f"src-{step}-{world}"
    write_source(
        src,
        total=TOTAL,
        step=step,
        weight_bounds=[0, 400, TOTAL],
        momentum_bounds=[0, 250, 800, TOTAL],
    )
    resp = client.post("/reshard", json={"new_world_size": world, "source_dir": str(src)})
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["status"] == "done", job
    assert job["step"] == step
    return job


def register_run(client, run_id, checkpoint_id, job_id):
    resp = client.post(
        "/runs",
        json={"run_id": run_id,
              "start_checkpoint": {"checkpoint_id": checkpoint_id, "job_id": job_id}},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def report(client, run_id, attempt, start, highest, produced=(), expect=200):
    body = {"attempt": attempt, "start_checkpoint": start, "highest_step": highest,
            "produced": [{"checkpoint_id": cid, "job_id": jid} for cid, jid in produced]}
    resp = client.post(f"/runs/{run_id}/reports", json=body)
    assert resp.status_code == expect, resp.text
    return resp


def reconcile(client, run_id, checkpoint_id, expect=200):
    resp = client.get(f"/runs/{run_id}/checkpoints/{checkpoint_id}/reconciliation")
    assert resp.status_code == expect, resp.text
    return resp.json() if expect == 200 else resp


@pytest.fixture()
def fork_run(client):
    """A run with a linear chain plus a fork (all steps derived from jobs):

        ckpt-100 --attempt 1--> ckpt-200 --attempt 2--> ckpt-350
            |------attempt 3--> ckpt-150        (fork, off the chain)
            |------attempt 4 (no product, highest 130)
    """
    jobs = {step: make_done_job(client, step) for step in (100, 150, 200, 350)}
    register_run(client, "run-1", "ckpt-100", jobs[100]["job_id"])
    report(client, "run-1", 1, "ckpt-100", 220, produced=[("ckpt-200", jobs[200]["job_id"])])
    report(client, "run-1", 2, "ckpt-200", 350, produced=[("ckpt-350", jobs[350]["job_id"])])
    report(client, "run-1", 3, "ckpt-100", 150, produced=[("ckpt-150", jobs[150]["job_id"])])
    report(client, "run-1", 4, "ckpt-100", 130)
    return jobs


class TestRunRegistration:
    def test_register_run_step_comes_from_job(self, client):
        job = make_done_job(client, 321)
        run = register_run(client, "r1", "ckpt-a", job["job_id"])
        assert run["run_id"] == "r1"
        assert run["checkpoints"] == [
            {"checkpoint_id": "ckpt-a", "job_id": job["job_id"],
             "step": 321, "produced_by": None}
        ]
        assert run["attempts"] == []

    def test_generated_run_id(self, client):
        job = make_done_job(client, 10)
        resp = client.post("/runs", json={
            "start_checkpoint": {"checkpoint_id": "c0", "job_id": job["job_id"]}})
        assert resp.status_code == 201
        assert resp.json()["run_id"].startswith("run-")

    def test_idempotent_reregistration(self, client):
        job = make_done_job(client, 10)
        first = register_run(client, "r1", "c0", job["job_id"])
        second = register_run(client, "r1", "c0", job["job_id"])
        assert first == second
        assert len(client.get("/runs").json()) == 1

    def test_conflicting_reregistration_409(self, client):
        job_a, job_b = make_done_job(client, 10), make_done_job(client, 20)
        register_run(client, "r1", "c0", job_a["job_id"])
        resp = client.post("/runs", json={
            "run_id": "r1",
            "start_checkpoint": {"checkpoint_id": "c0", "job_id": job_b["job_id"]}})
        assert resp.status_code == 409

    def test_unknown_job_422(self, client):
        resp = client.post("/runs", json={
            "run_id": "r1",
            "start_checkpoint": {"checkpoint_id": "c0", "job_id": "no-such-job"}})
        assert resp.status_code == 422

    def test_unfinished_job_422(self, client):
        job = make_done_job(client, 10)
        store = client.app.state.store
        stored = store.get(job["job_id"])
        stored.status = "failed"
        store.save(stored)
        resp = client.post("/runs", json={
            "run_id": "r1",
            "start_checkpoint": {"checkpoint_id": "c0", "job_id": job["job_id"]}})
        assert resp.status_code == 422
        assert "completed" in resp.json()["detail"]

    def test_bad_ids_422(self, client):
        job = make_done_job(client, 10)
        for bad in ("../evil", "a/b", " space ", "", ".hidden-ok-but.."):
            resp = client.post("/runs", json={
                "run_id": "r1",
                "start_checkpoint": {"checkpoint_id": bad, "job_id": job["job_id"]}})
            assert resp.status_code == 422, bad
        resp = client.post("/runs", json={
            "run_id": "../evil",
            "start_checkpoint": {"checkpoint_id": "c0", "job_id": job["job_id"]}})
        assert resp.status_code == 422

    def test_unknown_run_404(self, client):
        assert client.get("/runs/nope").status_code == 404
        resp = client.post("/runs/nope/reports", json={
            "attempt": 1, "start_checkpoint": "c0", "highest_step": 1})
        assert resp.status_code == 404
        assert client.get("/runs/nope/checkpoints/c0/reconciliation").status_code == 404


class TestReports:
    def test_start_must_be_registered(self, client, fork_run):
        resp = report(client, "run-1", 9, "ckpt-999", 500, expect=422)
        assert "not registered" in resp.json()["detail"]

    def test_highest_below_start_step_422(self, client, fork_run):
        report(client, "run-1", 9, "ckpt-200", 199, expect=422)

    def test_produced_must_reference_done_job(self, client, fork_run):
        report(client, "run-1", 9, "ckpt-100", 500,
               produced=[("ckpt-x", "no-such-job")], expect=422)
        pending = make_done_job(client, 400)
        store = client.app.state.store
        stored = store.get(pending["job_id"])
        stored.status = "failed"
        store.save(stored)
        resp = report(client, "run-1", 9, "ckpt-100", 500,
                      produced=[("ckpt-x", pending["job_id"])], expect=422)
        assert "completed" in resp.json()["detail"]

    def test_produced_step_must_exceed_start_step(self, client, fork_run):
        older = make_done_job(client, 100)  # same step as the start checkpoint
        resp = report(client, "run-1", 9, "ckpt-200", 500,
                      produced=[("ckpt-flat", older["job_id"])], expect=422)
        assert "out of bounds" in resp.json()["detail"]

    def test_produced_step_must_not_exceed_highest(self, client, fork_run):
        job_500 = make_done_job(client, 500)
        resp = report(client, "run-1", 9, "ckpt-100", 400,
                      produced=[("ckpt-500", job_500["job_id"])], expect=422)
        assert "out of bounds" in resp.json()["detail"]

    def test_duplicate_checkpoint_in_one_report_422(self, client, fork_run):
        job_500 = make_done_job(client, 500)
        report(client, "run-1", 9, "ckpt-100", 500,
               produced=[("ckpt-500", job_500["job_id"]),
                         ("ckpt-500", job_500["job_id"])], expect=422)

    def test_whole_report_rejected_atomically(self, client, fork_run):
        job_400, job_600 = make_done_job(client, 400), make_done_job(client, 600)
        # ckpt-ok is valid, ckpt-bad is out of bounds -> nothing may be applied.
        report(client, "run-1", 9, "ckpt-100", 500,
               produced=[("ckpt-ok", job_400["job_id"]),
                         ("ckpt-bad", job_600["job_id"])], expect=422)
        run = client.get("/runs/run-1").json()
        assert all(a["attempt"] != 9 for a in run["attempts"])
        assert all(c["checkpoint_id"] not in {"ckpt-ok", "ckpt-bad"}
                   for c in run["checkpoints"])

    def test_conflicting_attempt_start_409(self, client, fork_run):
        resp = report(client, "run-1", 1, "ckpt-200", 300, expect=409)
        assert "attempt 1" in resp.json()["detail"]

    def test_conflicting_checkpoint_attribution_409(self, client, fork_run):
        jobs = fork_run
        # Same checkpoint id, different producing attempt.
        resp = report(client, "run-1", 3, "ckpt-100", 250,
                      produced=[("ckpt-200", jobs[200]["job_id"])], expect=409)
        assert "ckpt-200" in resp.json()["detail"]
        # Same checkpoint id, different job behind it.
        other = make_done_job(client, 200, world=3)
        report(client, "run-1", 1, "ckpt-100", 220,
               produced=[("ckpt-200", other["job_id"])], expect=409)
        # The run start checkpoint cannot be re-claimed as a product.
        report(client, "run-1", 1, "ckpt-100", 220,
               produced=[("ckpt-100", jobs[100]["job_id"])], expect=409)

    def test_idempotent_replay(self, client, fork_run):
        jobs = fork_run
        before = client.get("/runs/run-1").json()
        report(client, "run-1", 1, "ckpt-100", 220,
               produced=[("ckpt-200", jobs[200]["job_id"])])
        after = client.get("/runs/run-1").json()
        attempt1 = [a for a in after["attempts"] if a["attempt"] == 1][0]
        assert attempt1["highest_step"] == 220
        assert attempt1["produced"] == ["ckpt-200"]  # not duplicated
        assert len(after["checkpoints"]) == len(before["checkpoints"])

    def test_late_report_cannot_lower_progress(self, client, fork_run):
        report(client, "run-1", 2, "ckpt-200", 250)  # late: 250 < recorded 350
        run = client.get("/runs/run-1").json()
        attempt2 = [a for a in run["attempts"] if a["attempt"] == 2][0]
        assert attempt2["highest_step"] == 350
        # ...but a higher report still raises it.
        report(client, "run-1", 2, "ckpt-200", 375)
        run = client.get("/runs/run-1").json()
        attempt2 = [a for a in run["attempts"] if a["attempt"] == 2][0]
        assert attempt2["highest_step"] == 375

    def test_late_report_may_register_new_checkpoints(self, client, fork_run):
        job_210 = make_done_job(client, 210)
        # Attempt 1 already recorded highest 220; this late report claims 210
        # and attaches a checkpoint at 210: valid, registers, progress stays.
        report(client, "run-1", 1, "ckpt-100", 210,
               produced=[("ckpt-210", job_210["job_id"])])
        run = client.get("/runs/run-1").json()
        attempt1 = [a for a in run["attempts"] if a["attempt"] == 1][0]
        assert attempt1["highest_step"] == 220
        assert "ckpt-210" in attempt1["produced"]
        ckpt = [c for c in run["checkpoints"] if c["checkpoint_id"] == "ckpt-210"][0]
        assert ckpt["step"] == 210 and ckpt["produced_by"] == 1


class TestReconciliation:
    def test_chain_kept_and_lost(self, client, fork_run):
        recon = reconcile(client, "run-1", "ckpt-350")
        assert recon["chain"] == ["ckpt-100", "ckpt-200", "ckpt-350"]
        assert recon["checkpoint_step"] == 350
        rows = {a["attempt"]: a for a in recon["attempts"]}
        # On-chain attempts: kept up to their product, overshoot lost.
        assert rows[1]["on_chain"] and rows[1]["kept_steps"] == 100
        assert rows[1]["lost_steps"] == 20  # 220 reported, 200 kept
        assert rows[2]["on_chain"] and rows[2]["kept_steps"] == 150
        assert rows[2]["lost_steps"] == 0
        # Off-chain attempts: everything lost; rerun steps counted per attempt.
        assert not rows[3]["on_chain"] and rows[3]["kept_steps"] == 0
        assert rows[3]["lost_steps"] == 50  # 101..150 rerun, counted for attempt 3
        assert rows[4]["lost_steps"] == 30
        assert recon["kept_total"] == 250
        assert recon["lost_total"] == 100

    def test_mid_chain_checkpoint_makes_later_attempts_off_chain(self, client, fork_run):
        recon = reconcile(client, "run-1", "ckpt-200")
        assert recon["chain"] == ["ckpt-100", "ckpt-200"]
        rows = {a["attempt"]: a for a in recon["attempts"]}
        assert rows[1]["kept_steps"] == 100 and rows[1]["lost_steps"] == 20
        assert rows[2]["on_chain"] is False
        assert rows[2]["kept_steps"] == 0 and rows[2]["lost_steps"] == 150
        assert recon["kept_total"] == 100
        assert recon["lost_total"] == 20 + 150 + 50 + 30

    def test_fork_product_chain(self, client, fork_run):
        recon = reconcile(client, "run-1", "ckpt-150")
        assert recon["chain"] == ["ckpt-100", "ckpt-150"]
        rows = {a["attempt"]: a for a in recon["attempts"]}
        assert rows[3]["on_chain"] and rows[3]["kept_steps"] == 50
        assert rows[1]["kept_steps"] == 0 and rows[1]["lost_steps"] == 120
        assert rows[2]["kept_steps"] == 0 and rows[2]["lost_steps"] == 150

    def test_run_start_checkpoint_keeps_nothing(self, client, fork_run):
        recon = reconcile(client, "run-1", "ckpt-100")
        assert recon["chain"] == ["ckpt-100"]
        assert recon["kept_total"] == 0
        assert recon["lost_total"] == 120 + 150 + 50 + 30
        for row in recon["attempts"]:
            assert row["on_chain"] is False and row["kept_steps"] == 0

    def test_unknown_checkpoint_404(self, client, fork_run):
        resp = reconcile(client, "run-1", "ckpt-nope", expect=404)
        assert "not registered" in resp.json()["detail"]

    def test_row_invariant(self, client, fork_run):
        # kept + lost == highest - start for every attempt, on any checkpoint.
        for ckpt in ("ckpt-100", "ckpt-150", "ckpt-200", "ckpt-350"):
            recon = reconcile(client, "run-1", ckpt)
            for row in recon["attempts"]:
                assert row["kept_steps"] + row["lost_steps"] == (
                    row["highest_step"] - row["start_step"]
                ), row


class TestConsistency:
    def test_state_survives_restart(self, client, fork_run):
        before = reconcile(client, "run-1", "ckpt-350")
        # A brand-new app over the same output root (service restart).
        app2 = create_app(
            Settings(
                source_dir=client.tmp_path / "no-default-source",
                output_root=client.tmp_path / "out",
                sync_execution=True,
            )
        )
        with TestClient(app2) as client2:
            run = client2.get("/runs/run-1")
            assert run.status_code == 200
            assert len(run.json()["checkpoints"]) == 4
            assert len(run.json()["attempts"]) == 4
            after = client2.get("/runs/run-1/checkpoints/ckpt-350/reconciliation").json()
        assert after == before

    def test_interleaved_reports_and_queries(self, client, fork_run):
        """Reports and queries racing each other: every response must reflect
        one committed state (row invariant holds, totals add up, no 5xx)."""
        errors: list[BaseException] = []
        stop = threading.Event()

        def reporter(attempt: int) -> None:
            try:
                for highest in range(150, 400, 25):
                    resp = client.post("/runs/run-1/reports", json={
                        "attempt": attempt, "start_checkpoint": "ckpt-100",
                        "highest_step": highest})
                    assert resp.status_code == 200, resp.text
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                if attempt == 9:
                    stop.set()

        def reader() -> None:
            try:
                while not stop.is_set():
                    run_resp = client.get("/runs/run-1")
                    assert run_resp.status_code == 200, run_resp.text
                    body = run_resp.json()
                    # The run view comes from one committed state: every
                    # attempt's start checkpoint resolves to a registered
                    # checkpoint, and produced ids are registered too.
                    ckpt_ids = {c["checkpoint_id"] for c in body["checkpoints"]}
                    for a in body["attempts"]:
                        assert a["start_checkpoint"] in ckpt_ids
                        assert set(a["produced"]) <= ckpt_ids
                    for ckpt in ("ckpt-100", "ckpt-200", "ckpt-350"):
                        resp = client.get(
                            f"/runs/run-1/checkpoints/{ckpt}/reconciliation")
                        assert resp.status_code == 200, resp.text
                        body = resp.json()
                        kept = sum(a["kept_steps"] for a in body["attempts"])
                        lost = sum(a["lost_steps"] for a in body["attempts"])
                        assert kept == body["kept_total"]
                        assert lost == body["lost_total"]
                        for row in body["attempts"]:
                            assert row["kept_steps"] + row["lost_steps"] == (
                                row["highest_step"] - row["start_step"]
                            ), row
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = ([threading.Thread(target=reporter, args=(n,)) for n in range(5, 10)]
                   + [threading.Thread(target=reader) for _ in range(3)])
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        run = client.get("/runs/run-1").json()
        assert {a["attempt"] for a in run["attempts"]} == {1, 2, 3, 4, 5, 6, 7, 8, 9}
        for n in range(5, 10):
            row = [a for a in run["attempts"] if a["attempt"] == n][0]
            assert row["highest_step"] == 375  # max of the reported series


class TestRunStoreUnit:
    def test_corrupt_run_dir_does_not_crash_startup(self, client, fork_run, tmp_path):
        run_dir = tmp_path / "out" / "runs" / "run-1"
        (run_dir / "run.json").write_text("{not json")
        store = RunStore(tmp_path / "out", client.app.state.store.get)
        assert store.get("run-1") is None  # skipped, service still boots
