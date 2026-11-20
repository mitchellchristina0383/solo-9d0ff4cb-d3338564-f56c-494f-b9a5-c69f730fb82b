"""Shared helpers: build deterministic source checkpoints in tmp dirs."""

from __future__ import annotations

import array
import json
import sys
from pathlib import Path


def floats_to_bytes(values) -> bytes:
    arr = array.array("f", values)
    if sys.byteorder == "big":
        arr.byteswap()
    return arr.tobytes()


def weight_value(i: int) -> float:
    return float(i)


def momentum_value(i: int) -> float:
    return -0.5 * i


def write_source(
    root: Path,
    total: int,
    step: int,
    weight_bounds: list[int],
    momentum_bounds: list[int],
) -> dict[str, bytes]:
    """Write a source checkpoint; each bounds list starts at 0 and ends at total.

    Returns {"weights": bytes, "momentum": bytes} — the full global tensors, so
    tests can assert rank outputs are exact byte slices of them.
    """
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"step": step, "total_length": total}
    globals_: dict[str, bytes] = {}
    for kind, bounds, value_at in (
        ("weights", weight_bounds, weight_value),
        ("momentum", momentum_bounds, momentum_value),
    ):
        assert bounds[0] == 0 and bounds[-1] == total
        (root / kind).mkdir(exist_ok=True)
        segments = []
        whole = bytearray()
        for idx, (start, end) in enumerate(zip(bounds, bounds[1:])):
            data = floats_to_bytes(value_at(i) for i in range(start, end))
            path = f"{kind}/part-{idx}.f32"
            (root / path).write_bytes(data)
            segments.append({"path": path, "start": start, "end": end})
            whole += data
        manifest[kind] = segments
        globals_[kind] = bytes(whole)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return globals_


def read_manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))
