import shutil

import pytest

pytest.importorskip("pyspark")
if shutil.which("java") is None:
    pytest.skip("Java not installed", allow_module_level=True)

from src import record_matching


def test_matching_finds_most_duplicates_with_few_false_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(record_matching, "INTERIM_DIR", tmp_path)
    metrics = record_matching.main(n_entities=800, reports_dir=tmp_path)
    assert metrics["precision"] > 0.9
    assert metrics["recall"] > 0.85
    assert metrics["comparisons_saved_pct"] > 90
