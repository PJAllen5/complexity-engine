"""Keeps the fast test run honest: every known-answer validation file carries the validation
marker, so `pytest -m "not validation"` skips all of the slow ones and only those."""

from pathlib import Path

TESTS = Path(__file__).resolve().parent
# Fast reference-implementation checks that run with the unit tests despite the name.
NOT_SLOW = {"test_metric_validation.py"}


def test_every_validation_file_is_marked():
    files = {*TESTS.glob("test_*_validation.py"), *TESTS.glob("test_*_synthetic.py")}
    unmarked = sorted(f.name for f in files if f.name not in NOT_SLOW and "pytest.mark.validation" not in f.read_text())
    assert not unmarked, f"Add `pytest.mark.validation` to pytestmark in: {', '.join(unmarked)}"


def test_no_other_file_is_marked():
    marked = {f.name for f in TESTS.glob("test_*.py") if "pytest.mark.validation" in f.read_text()}
    expected = {f.name for f in (*TESTS.glob("test_*_validation.py"), *TESTS.glob("test_*_synthetic.py"))} - NOT_SLOW
    assert marked - {"test_markers.py"} == expected
