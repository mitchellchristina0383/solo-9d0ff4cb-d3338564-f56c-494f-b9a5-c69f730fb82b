"""Training-step reconciliation: runs, attempts and produced checkpoints.

When training is rerun from an old checkpoint, the on-call operator needs to
know how many steps of each attempt are still reflected in a given (selected)
checkpoint.  This module tracks just enough to answer that:

- A **run** is registered with a start checkpoint.  Every checkpoint
  references a *completed* reshard job; the checkpoint's step is the job's
  manifest step (single source of truth — never supplied by the client).
- The scheduler **reports** an attempt: attempt number, the checkpoint it
  started from, its highest completed step, and optionally the checkpoints it
  produced.  A report is validated against the committed state and committed
  atomically — any violation rejects the whole report.
- Attempts may only start from checkpoints already registered in the run.
- A late report can never lower the recorded highest step (a max is kept).
- A produced checkpoint's step must be greater than the attempt's start step
  and must not exceed the report's highest completed step.
- Conflicting start points or produced-checkpoint attribution, or
  out-of-bounds produced steps, reject the whole report.
- **Reconciliation** for any registered checkpoint walks the actual produced
  chain back to the run start: on-chain attempts contribute kept steps up to
  the checkpoint they produced on the chain; everything else — including all
  steps of off-chain attempts — is lost.  The same step rerun by several
  attempts is counted per attempt, separately.

State is persisted to ``<output_root>/runs/<run_id>/run.json`` (atomic
replace, like job.json) and every public store operation runs under one lock,
so an interleaved report and query always observe a single committed state.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .jobs import Job

RUNS_DIR_NAME = "runs"
RUN_META_NAME = "run.json"

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunError(Exception):
    """Base class for run/reconciliation failures."""


class UnknownRunError(RunError):
    """No run with this id."""


class UnknownCheckpointError(RunError):
    """The checkpoint is not registered in this run."""


class ReportConflictError(RunError):
    """The report contradicts already-committed records (409)."""


class ReportInvalidError(RunError):
    """The report references invalid objects or is out of bounds (422)."""


def _validate_id(value: str, kind: str) -> str:
    """Ids become directory names (run_id) or JSON keys — keep them tame."""
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ReportInvalidError(
            f"{kind} {value!r} is invalid: use 1-128 chars from "
            "[A-Za-z0-9._-], starting with a letter or digit"
        )
    return value


@dataclass
class Checkpoint:
    checkpoint_id: str
    job_id: str            # the completed reshard job this checkpoint points at
    step: int              # taken from the job's manifest step
    produced_by: int | None  # attempt that produced it; None = run start

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "job_id": self.job_id,
            "step": self.step,
            "produced_by": self.produced_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        produced_by = data.get("produced_by")
        return cls(
            checkpoint_id=str(data["checkpoint_id"]),
            job_id=str(data["job_id"]),
            step=int(data["step"]),
            produced_by=None if produced_by is None else int(produced_by),
        )


@dataclass
class Attempt:
    attempt: int
    start_checkpoint: str          # checkpoint_id the attempt started from
    highest_step: int              # max over committed reports; never decreases
    produced: list[str] = field(default_factory=list)  # checkpoint ids, in order

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "start_checkpoint": self.start_checkpoint,
            "highest_step": self.highest_step,
            "produced": list(self.produced),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Attempt":
        return cls(
            attempt=int(data["attempt"]),
            start_checkpoint=str(data["start_checkpoint"]),
            highest_step=int(data["highest_step"]),
            produced=[str(p) for p in data.get("produced", [])],
        )


@dataclass
class Run:
    run_id: str
    start_checkpoint: str                 # checkpoint_id of the initial checkpoint
    checkpoints: dict[str, Checkpoint]    # registration order is preserved
    attempts: dict[int, Attempt]
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "start_checkpoint": self.start_checkpoint,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "checkpoints": [c.to_dict() for c in self.checkpoints.values()],
            "attempts": [a.to_dict() for a in self.sorted_attempts()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Run":
        checkpoints = [Checkpoint.from_dict(c) for c in data.get("checkpoints", [])]
        attempts = [Attempt.from_dict(a) for a in data.get("attempts", [])]
        return cls(
            run_id=str(data["run_id"]),
            start_checkpoint=str(data["start_checkpoint"]),
            checkpoints={c.checkpoint_id: c for c in checkpoints},
            attempts={a.attempt: a for a in attempts},
            created_at=str(data.get("created_at", _utcnow())),
            updated_at=str(data.get("updated_at", _utcnow())),
        )

    def sorted_attempts(self) -> list[Attempt]:
        return [self.attempts[n] for n in sorted(self.attempts)]


@dataclass(frozen=True)
class ProducedRef:
    """One produced checkpoint as carried by a scheduler report."""

    checkpoint_id: str
    job_id: str


@dataclass(frozen=True)
class AttemptRecon:
    """Per-attempt reconciliation row: steps kept in / lost from the selected
    checkpoint.  kept_steps + lost_steps == highest_step - start_step."""

    attempt: int
    on_chain: bool
    start_checkpoint: str
    start_step: int
    highest_step: int
    kept_steps: int
    lost_steps: int


@dataclass(frozen=True)
class Reconciliation:
    run_id: str
    checkpoint_id: str
    checkpoint_step: int
    chain: list[str]  # checkpoint ids from the run start to the selected one
    attempts: list[AttemptRecon]
    kept_total: int
    lost_total: int


def reconcile_run(run: Run, checkpoint_id: str) -> Reconciliation:
    """Reconcile every attempt of ``run`` against the selected checkpoint.

    Pure function over a run snapshot: the caller (store) supplies a run read
    from one committed state, so interleaved reports cannot tear the result.
    """
    checkpoint = run.checkpoints.get(checkpoint_id)
    if checkpoint is None:
        raise UnknownCheckpointError(
            f"checkpoint {checkpoint_id!r} is not registered in run {run.run_id!r}"
        )

    # Walk the actual produced chain backwards: each checkpoint names the
    # attempt that produced it, each attempt names the checkpoint it started
    # from.  Steps strictly increase along produced links, so this terminates
    # at the run start (produced_by=None).
    chain: list[str] = []
    chain_link: dict[int, str] = {}  # attempt no -> checkpoint it produced on this chain
    cursor = checkpoint_id
    while True:
        cp = run.checkpoints[cursor]
        chain.append(cursor)
        if cp.produced_by is None:
            break
        chain_link[cp.produced_by] = cursor
        cursor = run.attempts[cp.produced_by].start_checkpoint
    chain.reverse()

    rows: list[AttemptRecon] = []
    for attempt in run.sorted_attempts():
        start_step = run.checkpoints[attempt.start_checkpoint].step
        kept_until = chain_link.get(attempt.attempt)
        if kept_until is not None:
            # On the chain: steps up to the checkpoint this attempt produced
            # are kept in the selected checkpoint; anything completed beyond
            # it was discarded.
            kept_step = run.checkpoints[kept_until].step
            kept = kept_step - start_step
            lost = attempt.highest_step - kept_step
        else:
            # Off the chain: none of this attempt's steps survive in the
            # selected checkpoint.  Rerun steps are counted per attempt, so
            # no de-duplication against other attempts happens here.
            kept = 0
            lost = attempt.highest_step - start_step
        rows.append(
            AttemptRecon(
                attempt=attempt.attempt,
                on_chain=kept_until is not None,
                start_checkpoint=attempt.start_checkpoint,
                start_step=start_step,
                highest_step=attempt.highest_step,
                kept_steps=kept,
                lost_steps=lost,
            )
        )
    return Reconciliation(
        run_id=run.run_id,
        checkpoint_id=checkpoint_id,
        checkpoint_step=checkpoint.step,
        chain=chain,
        attempts=rows,
        kept_total=sum(r.kept_steps for r in rows),
        lost_total=sum(r.lost_steps for r in rows),
    )


def _references_ok(run: Run) -> bool:
    """Referential integrity of a loaded run: every pointer must resolve."""
    if run.start_checkpoint not in run.checkpoints:
        return False
    for cp in run.checkpoints.values():
        if cp.produced_by is not None and cp.produced_by not in run.attempts:
            return False
    for attempt in run.attempts.values():
        if attempt.start_checkpoint not in run.checkpoints:
            return False
        if any(p not in run.checkpoints for p in attempt.produced):
            return False
    return True


class RunStore:
    """Loads, commits and persists runs under ``<output_root>/runs/``.

    A single lock serialises validation+commit of reports and every read, so
    each API call observes exactly one committed state.  ``job_lookup``
    resolves reshard jobs (checkpoint references must point at done jobs).
    """

    def __init__(self, output_root: Path, job_lookup: Callable[[str], Job | None]):
        self.runs_root = Path(output_root) / RUNS_DIR_NAME
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._job_lookup = job_lookup
        self._lock = threading.Lock()
        self._runs: dict[str, Run] = {}
        self._load_all()

    # ------------------------------------------------------------- persistence

    def _load_all(self) -> None:
        for entry in sorted(self.runs_root.iterdir()):
            meta = entry / RUN_META_NAME
            if not entry.is_dir() or not meta.is_file():
                continue
            try:
                run = Run.from_dict(json.loads(meta.read_text(encoding="utf-8")))
            except (OSError, ValueError, KeyError):
                continue  # corrupt run dir: leave it alone, don't crash startup
            if not _references_ok(run):
                continue  # dangling references (hand-edited?): skip, don't crash
            self._runs[run.run_id] = run

    def _save(self, run: Run) -> None:
        run.updated_at = _utcnow()
        run_dir = self.runs_root / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(run.to_dict(), indent=2) + "\n"
        # Atomic replace so a crash never leaves a half-written run.json.
        fd, tmp = tempfile.mkstemp(dir=run_dir, prefix=".run-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, run_dir / RUN_META_NAME)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # ----------------------------------------------------------------- lookups

    @staticmethod
    def _copy(run: Run) -> Run:
        """Deep copy via the dict form — callers can never mutate store state."""
        return Run.from_dict(run.to_dict())

    def get(self, run_id: str) -> Run | None:
        """A snapshot of the run as of one committed state (or None)."""
        with self._lock:
            run = self._runs.get(run_id)
            return None if run is None else self._copy(run)

    def all(self) -> list[Run]:
        """Snapshots of all runs as of one committed state."""
        with self._lock:
            return [self._copy(r)
                    for r in sorted(self._runs.values(), key=lambda r: r.created_at)]

    def _done_job_step(self, job_id: str, what: str) -> int:
        """Resolve a checkpoint's job reference: the job must exist and be a
        completed reshard; its manifest step becomes the checkpoint's step."""
        job = self._job_lookup(job_id)
        if job is None:
            raise ReportInvalidError(f"{what}: unknown reshard job {job_id!r}")
        if job.status != "done":
            raise ReportInvalidError(
                f"{what}: reshard job {job_id!r} is {job.status}; "
                "a checkpoint must reference a completed reshard job"
            )
        return job.step

    # --------------------------------------------------------------- mutations

    def register_run(self, run_id: str | None, start_checkpoint_id: str,
                     job_id: str) -> Run:
        """Register a run with its start checkpoint.  Re-registering the same
        run with the same start checkpoint is idempotent; a different start
        checkpoint for an existing run id is a conflict."""
        _validate_id(start_checkpoint_id, "checkpoint_id")
        step = self._done_job_step(job_id, "start checkpoint")
        with self._lock:
            if run_id is None:
                run_id = f"run-{uuid.uuid4().hex[:12]}"
            else:
                _validate_id(run_id, "run_id")
            existing = self._runs.get(run_id)
            if existing is not None:
                start = existing.checkpoints[existing.start_checkpoint]
                if start.checkpoint_id == start_checkpoint_id and start.job_id == job_id:
                    return self._copy(existing)  # idempotent retry of the same registration
                raise ReportConflictError(
                    f"run {run_id!r} already exists with start checkpoint "
                    f"{start.checkpoint_id!r} (job {start.job_id!r})"
                )
            run = Run(
                run_id=run_id,
                start_checkpoint=start_checkpoint_id,
                checkpoints={
                    start_checkpoint_id: Checkpoint(
                        checkpoint_id=start_checkpoint_id,
                        job_id=job_id,
                        step=step,
                        produced_by=None,
                    )
                },
                attempts={},
            )
            self._runs[run_id] = run
            self._save(run)
            return self._copy(run)

    def commit_report(self, run_id: str, attempt_no: int, start_checkpoint: str,
                      highest_step: int, produced: list[ProducedRef]) -> Run:
        """Validate a scheduler report against the committed state and commit
        it atomically.  Any violation rejects the whole report — no partial
        attempt updates, no partial checkpoint registration."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise UnknownRunError(f"unknown run {run_id!r}")

            start = run.checkpoints.get(start_checkpoint)
            if start is None:
                raise ReportInvalidError(
                    f"attempt {attempt_no} starts from {start_checkpoint!r}, which is "
                    f"not registered in run {run_id!r}; attempts may only start "
                    "from checkpoints registered in this run"
                )
            if highest_step < start.step:
                raise ReportInvalidError(
                    f"highest_step {highest_step} is below the start checkpoint's "
                    f"step {start.step}"
                )

            attempt = run.attempts.get(attempt_no)
            if attempt is not None and attempt.start_checkpoint != start_checkpoint:
                raise ReportConflictError(
                    f"attempt {attempt_no} already recorded as starting from "
                    f"{attempt.start_checkpoint!r}, cannot restart from "
                    f"{start_checkpoint!r}"
                )

            # Validate every produced checkpoint before touching anything.
            seen: set[str] = set()
            new_checkpoints: list[Checkpoint] = []
            for ref in produced:
                _validate_id(ref.checkpoint_id, "checkpoint_id")
                if ref.checkpoint_id in seen:
                    raise ReportInvalidError(
                        f"checkpoint {ref.checkpoint_id!r} listed twice in one report"
                    )
                seen.add(ref.checkpoint_id)
                step = self._done_job_step(ref.job_id,
                                           f"produced checkpoint {ref.checkpoint_id!r}")
                existing = run.checkpoints.get(ref.checkpoint_id)
                if existing is not None:
                    if existing.job_id == ref.job_id and existing.produced_by == attempt_no:
                        continue  # idempotent re-report of the same checkpoint
                    raise ReportConflictError(
                        f"checkpoint {ref.checkpoint_id!r} is already registered "
                        f"(job {existing.job_id!r}, produced by attempt "
                        f"{existing.produced_by}); cannot attribute it to attempt "
                        f"{attempt_no} with job {ref.job_id!r}"
                    )
                if not (start.step < step <= highest_step):
                    raise ReportInvalidError(
                        f"produced checkpoint {ref.checkpoint_id!r} has step {step}, "
                        f"out of bounds: must be greater than the start step "
                        f"{start.step} and at most the reported highest completed "
                        f"step {highest_step}"
                    )
                new_checkpoints.append(
                    Checkpoint(
                        checkpoint_id=ref.checkpoint_id,
                        job_id=ref.job_id,
                        step=step,
                        produced_by=attempt_no,
                    )
                )

            # Commit: a late report can only raise the recorded highest step.
            if attempt is None:
                attempt = Attempt(
                    attempt=attempt_no,
                    start_checkpoint=start_checkpoint,
                    highest_step=highest_step,
                )
                run.attempts[attempt_no] = attempt
            else:
                attempt.highest_step = max(attempt.highest_step, highest_step)
            for cp in new_checkpoints:
                run.checkpoints[cp.checkpoint_id] = cp
                attempt.produced.append(cp.checkpoint_id)
            self._save(run)
            return self._copy(run)

    # ------------------------------------------------------------------ query

    def reconcile(self, run_id: str, checkpoint_id: str) -> Reconciliation:
        """Reconcile attempts against the selected checkpoint, computed from
        one committed state (the lock is held for the whole computation)."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise UnknownRunError(f"unknown run {run_id!r}")
            return reconcile_run(run, checkpoint_id)
