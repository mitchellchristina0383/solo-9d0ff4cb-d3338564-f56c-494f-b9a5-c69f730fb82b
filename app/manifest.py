"""Manifest loading and validation.

The source checkpoint lives on a read-only volume and is described by a
``manifest.json``::

    {
      "step": 12000,
      "total_length": 1000,
      "weights":  [{"path": "weights/part-0.f32", "start": 0, "end": 400}, ...],
      "momentum": [{"path": "momentum/part-0.f32", "start": 0, "end": 250}, ...]
    }

Each segment states which half-open global element interval ``[start, end)``
of the flattened float32 tensor its file holds.  Weights and momentum may use
different split points.  For the conversion to be sound, the segments of each
tensor must tile ``[0, total_length)`` exactly: no gaps, no overlaps, and every
file must be exactly ``(end - start) * 4`` bytes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

FLOAT32_BYTES = 4
MANIFEST_NAME = "manifest.json"


class ManifestError(ValueError):
    """Raised when the source manifest or the files it describes are invalid."""


@dataclass(frozen=True)
class Segment:
    path: str  # relative to the source root
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Manifest:
    step: int
    total_length: int
    weights: tuple[Segment, ...]
    momentum: tuple[Segment, ...]


def _parse_segments(raw: Any, kind: str, total: int) -> tuple[Segment, ...]:
    if not isinstance(raw, list) or not raw:
        raise ManifestError(f"{kind}: must be a non-empty list of segments")

    segments: list[Segment] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ManifestError(f"{kind}[{i}]: segment must be an object")
        try:
            path = item["path"]
            start = int(item["start"])
            end = int(item["end"])
        except KeyError as exc:
            raise ManifestError(f"{kind}[{i}]: missing field {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"{kind}[{i}]: start/end must be integers") from exc

        if not isinstance(path, str) or not path:
            raise ManifestError(f"{kind}[{i}]: path must be a non-empty string")
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts:
            raise ManifestError(f"{kind}[{i}]: path {path!r} must stay inside the source directory")
        if not (0 <= start < end <= total):
            raise ManifestError(
                f"{kind}[{i}]: interval [{start}, {end}) is empty or outside [0, {total})"
            )
        segments.append(Segment(path=path, start=start, end=end))

    # The segments must tile [0, total) exactly.  Sorting by start lets a
    # single linear pass detect both gaps and overlaps.
    segments.sort(key=lambda s: (s.start, s.end))
    expect = 0
    for seg in segments:
        if seg.start < expect:
            raise ManifestError(
                f"{kind}: overlap — segment [{seg.start}, {seg.end}) starts before "
                f"index {expect} which is already covered"
            )
        if seg.start > expect:
            raise ManifestError(
                f"{kind}: gap — no segment covers [{expect}, {seg.start})"
            )
        expect = seg.end
    if expect != total:
        raise ManifestError(
            f"{kind}: coverage stops at {expect}, but total_length is {total} "
            f"(gap at the tail)"
        )
    return tuple(segments)


def parse_manifest(data: Any) -> Manifest:
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be a JSON object")
    try:
        step = int(data["step"])
        total = int(data["total_length"])
    except KeyError as exc:
        raise ManifestError(f"missing field {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ManifestError("step and total_length must be integers") from exc
    if step < 0:
        raise ManifestError("step must be >= 0")
    if total <= 0:
        raise ManifestError("total_length must be > 0")
    return Manifest(
        step=step,
        total_length=total,
        weights=_parse_segments(data.get("weights"), "weights", total),
        momentum=_parse_segments(data.get("momentum"), "momentum", total),
    )


def _check_files(source_dir: Path, segments: tuple[Segment, ...], kind: str) -> None:
    for seg in segments:
        full = source_dir / seg.path
        try:
            size = os.path.getsize(full)
        except OSError as exc:
            raise ManifestError(f"{kind}: cannot stat {seg.path!r}: {exc.strerror or exc}") from exc
        expected = seg.length * FLOAT32_BYTES
        if size != expected:
            raise ManifestError(
                f"{kind}: {seg.path!r} is {size} bytes, expected {expected} "
                f"({seg.length} float32 elements)"
            )


def load_manifest(source_dir: Path) -> Manifest:
    """Read, parse and fully validate the manifest under ``source_dir``."""
    manifest_path = source_dir / MANIFEST_NAME
    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"cannot read {manifest_path}: {exc.strerror or exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{MANIFEST_NAME} is not valid JSON: {exc}") from exc
    manifest = parse_manifest(data)
    _check_files(source_dir, manifest.weights, "weights")
    _check_files(source_dir, manifest.momentum, "momentum")
    return manifest
