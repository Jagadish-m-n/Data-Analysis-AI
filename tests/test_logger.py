"""Tests for app/logger/custom_logger.py.

Run with:  uv run pytest tests/test_logger.py -v
"""

import json
import os

import pytest

from app.logger.custom_logger import LOG_FILE, get_logger

logger = get_logger(__name__)


def last_log(caplog) -> dict:
    """The last line this file logged, parsed from JSON."""
    return json.loads(caplog.records[-1].getMessage())


def test_log_line_is_json_with_fields(caplog):
    # Under pytest, basicConfig in custom_logger is skipped (pytest's handler is already on
    # the root logger), so the level stays WARNING unless set here
    caplog.set_level("INFO")
    logger.info("file_uploaded", filename="sales.csv", rows=120)

    log = last_log(caplog)
    assert log["event"] == "file_uploaded"
    assert log["filename"] == "sales.csv"
    assert log["rows"] == 120
    assert log["level"] == "info"
    assert "timestamp" in log


def test_warning_and_error_levels(caplog):
    logger.warning("missing_values_detected", column="age", count=15)
    assert last_log(caplog)["level"] == "warning"

    logger.error("analysis_failed", column="age", reason="Invalid data")
    assert last_log(caplog)["level"] == "error"


def test_exception_includes_traceback(caplog):
    try:
        10 / 0
    except ZeroDivisionError:
        logger.exception("division_error_occurred", operation="division")

    log = last_log(caplog)
    assert log["event"] == "division_error_occurred"
    assert "ZeroDivisionError" in log["exception"]


def test_log_file_is_created():
    assert os.path.isfile(LOG_FILE)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
