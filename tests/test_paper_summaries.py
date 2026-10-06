import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("release_table_rebuild", Path(__file__).resolve().parents[1] / "scripts/rebuild_tables.py")
tables = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tables)


def test_all_published_cells_from_individual_values():
    rows = tables.rebuild()
    assert len(rows) == 561
    assert len([r for r in rows if r["table"] == "synthetic_prediction"]) == 72
    assert len([r for r in rows if r["table"] == "synthetic_recovery"]) == 48
    assert len([r for r in rows if r["table"] == "hiv_nrti"]) == 96
    assert all(r["n"] == 5 for r in rows)
    assert all(r["method"] != "DIPS-PATH" for r in rows if r["table"] == "synthetic_recovery")


@pytest.mark.parametrize("values", [[1, 2, 3, 4], [1, 2, 3, 4, float("nan")]])
def test_incomplete_or_nonfinite_seed_values_are_rejected(values):
    with pytest.raises(ValueError):
        tables.five(values, {"mean": 3, "sample_sd": 1})


def test_stale_reference_number_is_rejected():
    with pytest.raises(ValueError, match="mismatch"):
        tables.five([1, 2, 3, 4, 5], {"mean": 4, "sample_sd": 1})


def test_sd_is_sample_sd_not_population_sd():
    with pytest.raises(ValueError, match="mismatch"):
        tables.five([1, 2, 3, 4, 5], {"mean": 3, "sample_sd": 2**0.5})
