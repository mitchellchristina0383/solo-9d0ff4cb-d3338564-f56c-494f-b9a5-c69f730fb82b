import pytest

from app.manifest import Segment
from app.plans import plan_copy_ops, rank_intervals


class TestRankIntervals:
    def test_even_split_exact(self):
        assert rank_intervals(12, 3) == [(0, 4), (4, 8), (8, 12)]

    def test_uneven_split_covers_exactly_once(self):
        intervals = rank_intervals(10007, 5)
        assert intervals[0][0] == 0
        assert intervals[-1][1] == 10007
        for (a_start, a_end), (b_start, b_end) in zip(intervals, intervals[1:]):
            assert a_end == b_start
        sizes = [e - s for s, e in intervals]
        assert max(sizes) - min(sizes) <= 1  # near-even

    def test_world_size_one(self):
        assert rank_intervals(7, 1) == [(0, 7)]

    def test_world_size_exceeding_total_rejected(self):
        with pytest.raises(ValueError, match="exceeds total_length"):
            rank_intervals(3, 4)

    def test_zero_world_size_rejected(self):
        with pytest.raises(ValueError):
            rank_intervals(10, 0)


class TestCopyOps:
    def test_single_source_segment(self):
        segs = (Segment("w/part-0.f32", 0, 100),)
        ops = plan_copy_ops(segs, 10, 30)
        got = [(o.path, o.src_offset, o.dst_offset, o.length) for o in ops]
        assert got == [("w/part-0.f32", 10, 0, 20)]

    def test_spanning_multiple_segments(self):
        segs = (
            Segment("w/part-0.f32", 0, 40),
            Segment("w/part-1.f32", 40, 75),
            Segment("w/part-2.f32", 75, 100),
        )
        ops = plan_copy_ops(segs, 30, 80)
        got = [(o.path, o.src_offset, o.dst_offset, o.length) for o in ops]
        assert got == [
            ("w/part-0.f32", 30, 0, 10),
            ("w/part-1.f32", 0, 10, 35),
            ("w/part-2.f32", 0, 45, 5),
        ]
        assert sum(o.length for o in ops) == 50

    def test_dst_offsets_are_contiguous(self):
        segs = (
            Segment("w/a.f32", 0, 3),
            Segment("w/b.f32", 3, 9),
            Segment("w/c.f32", 9, 10),
        )
        ops = plan_copy_ops(segs, 0, 10)
        cursor = 0
        for op in ops:
            assert op.dst_offset == cursor
            cursor += op.length
        assert cursor == 10
