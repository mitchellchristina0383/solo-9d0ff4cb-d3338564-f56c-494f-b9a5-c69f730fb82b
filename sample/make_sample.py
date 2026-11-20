#!/usr/bin/env python3
"""Generate a deterministic sample source checkpoint.

Layout produced under DEST (defaults to ./sample/source):

    manifest.json
    weights/part-0.f32  ...   raw little-endian float32, uneven splits
    momentum/part-0.f32 ...   different split points than weights

Values are a verifiable pattern: weights[i] = float(i), momentum[i] = -0.5*i.
"""

from __future__ import annotations

import argparse
import array
import json
import sys
from pathlib import Path

DEFAULT_TOTAL = 10007  # prime on purpose: exercises uneven rank splits
DEFAULT_STEP = 7459
WEIGHT_SPLITS = (0, 3500, 7200)      # boundaries; final end = total
MOMENTUM_SPLITS = (0, 2000, 9000)    # deliberately different from weights


def write_parts(dest: Path, subdir: str, splits: tuple[int, ...], total: int,
                value_at) -> list[dict]:
    (dest / subdir).mkdir(parents=True, exist_ok=True)
    segments = []
    bounds = list(splits) + [total]
    for idx, (start, end) in enumerate(zip(bounds, bounds[1:])):
        vals = array.array("f", (value_at(i) for i in range(start, end)))
        if sys.byteorder == "big":
            vals.byteswap()
        path = f"{subdir}/part-{idx}.f32"
        (dest / path).write_bytes(vals.tobytes())
        segments.append({"path": path, "start": start, "end": end})
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", nargs="?", default=str(Path(__file__).parent / "source"))
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {
        "step": args.step,
        "total_length": args.total,
        "weights": write_parts(dest, "weights", WEIGHT_SPLITS, args.total, float),
        "momentum": write_parts(dest, "momentum", MOMENTUM_SPLITS, args.total,
                                lambda i: -0.5 * i),
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"sample checkpoint written to {dest} "
          f"(step={args.step}, total_length={args.total})")


if __name__ == "__main__":
    main()
