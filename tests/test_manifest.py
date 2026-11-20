import json

import pytest

from app.manifest import ManifestError, load_manifest
from tests.conftest import floats_to_bytes, write_source


def make_valid(tmp_path):
    return write_source(
        tmp_path / "src",
        total=100,
        step=42,
        weight_bounds=[0, 40, 100],
        momentum_bounds=[0, 55, 100],
    )


class TestValidManifest:
    def test_loads(self, tmp_path):
        make_valid(tmp_path)
        m = load_manifest(tmp_path / "src")
        assert m.step == 42
        assert m.total_length == 100
        assert [(s.start, s.end) for s in m.weights] == [(0, 40), (40, 100)]
        assert [(s.start, s.end) for s in m.momentum] == [(0, 55), (55, 100)]

    def test_unsorted_segments_accepted_after_sort(self, tmp_path):
        make_valid(tmp_path)
        src = tmp_path / "src"
        manifest = json.loads((src / "manifest.json").read_text())
        manifest["weights"] = list(reversed(manifest["weights"]))
        (src / "manifest.json").write_text(json.dumps(manifest))
        m = load_manifest(src)
        assert [(s.start, s.end) for s in m.weights] == [(0, 40), (40, 100)]


class TestGapOverlapRejection:
    def _break(self, tmp_path, mutate):
        make_valid(tmp_path)
        src = tmp_path / "src"
        manifest = json.loads((src / "manifest.json").read_text())
        mutate(manifest)
        (src / "manifest.json").write_text(json.dumps(manifest))
        return src

    def test_gap_in_weights_rejected(self, tmp_path):
        src = self._break(tmp_path, lambda m: m["weights"][1].update(start=41))
        with pytest.raises(ManifestError, match="gap"):
            load_manifest(src)

    def test_overlap_in_weights_rejected(self, tmp_path):
        src = self._break(tmp_path, lambda m: m["weights"][1].update(start=39))
        with pytest.raises(ManifestError, match="overlap"):
            load_manifest(src)

    def test_tail_gap_rejected(self, tmp_path):
        src = self._break(tmp_path, lambda m: m["momentum"][1].update(end=99))
        with pytest.raises(ManifestError, match="coverage stops"):
            load_manifest(src)

    def test_interval_out_of_range_rejected(self, tmp_path):
        src = self._break(tmp_path, lambda m: m["weights"][1].update(end=101))
        with pytest.raises(ManifestError, match="outside"):
            load_manifest(src)

    def test_empty_interval_rejected(self, tmp_path):
        src = self._break(tmp_path, lambda m: m["weights"][0].update(end=0))
        with pytest.raises(ManifestError, match="outside"):
            load_manifest(src)


class TestFileChecks:
    def test_missing_file_rejected(self, tmp_path):
        make_valid(tmp_path)
        (tmp_path / "src" / "weights" / "part-1.f32").unlink()
        with pytest.raises(ManifestError, match="cannot stat"):
            load_manifest(tmp_path / "src")

    def test_wrong_file_size_rejected(self, tmp_path):
        make_valid(tmp_path)
        (tmp_path / "src" / "momentum" / "part-0.f32").write_bytes(floats_to_bytes([0.0] * 54))
        with pytest.raises(ManifestError, match="expected"):
            load_manifest(tmp_path / "src")

    def test_path_traversal_rejected(self, tmp_path):
        make_valid(tmp_path)
        src = tmp_path / "src"
        manifest = json.loads((src / "manifest.json").read_text())
        manifest["weights"][0]["path"] = "../escape.f32"
        (src / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ManifestError, match="inside"):
            load_manifest(src)
