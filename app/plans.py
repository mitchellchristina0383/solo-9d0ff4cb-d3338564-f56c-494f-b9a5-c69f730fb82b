"""Split math: even global-index ranges per rank and per-rank copy plans."""

from __future__ import annotations

from dataclasses import dataclass

from .manifest import Segment


def rank_intervals(total: int, world_size: int) -> list[tuple[int, int]]:
    """Split ``[0, total)`` into ``world_size`` contiguous, near-even ranges.

    Rank ``i`` owns ``[i*total//W, (i+1)*total//W)`` — sizes differ by at most
    one element and every index is covered exactly once.
    """
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if world_size > total:
        raise ValueError(
            f"world_size {world_size} exceeds total_length {total}; "
            "every rank must own at least one element"
        )
    return [(i * total // world_size, (i + 1) * total // world_size) for i in range(world_size)]


@dataclass(frozen=True)
class CopyOp:
    """Copy ``length`` elements from ``path`` starting at ``src_offset`` into
    the output at ``dst_offset`` (all offsets in float32 elements)."""

    path: str
    src_offset: int
    dst_offset: int
    length: int


def plan_copy_ops(segments: tuple[Segment, ...], start: int, end: int) -> list[CopyOp]:
    """Gather the source reads needed to materialise ``[start, end)``.

    ``segments`` must tile the whole index space (validated upstream), so the
    resulting ops cover ``[start, end)`` exactly and in order.
    """
    ops: list[CopyOp] = []
    cursor = start
    for seg in segments:
        ov_start = max(seg.start, start)
        ov_end = min(seg.end, end)
        if ov_start >= ov_end:
            continue
        if ov_start != cursor:  # defensive: validated manifests never trigger this
            raise ValueError(f"source segments do not cover index {cursor}")
        ops.append(
            CopyOp(
                path=seg.path,
                src_offset=ov_start - seg.start,
                dst_offset=ov_start - start,
                length=ov_end - ov_start,
            )
        )
        cursor = ov_end
    if cursor != end:
        raise ValueError(f"source segments cover up to {cursor}, need {end}")
    return ops
