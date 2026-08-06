import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import settings
from src.security.validator import InputValidator


def test_validate_query_truncates_long_input():
    long_query = "a" * (settings.MAX_QUERY_LENGTH + 500)
    result = InputValidator.validate_query(long_query)
    assert len(result) == settings.MAX_QUERY_LENGTH
    assert result == "a" * settings.MAX_QUERY_LENGTH


def test_validate_query_rejects_empty():
    try:
        InputValidator.validate_query("   ")
        assert False, "expected ValueError"
    except ValueError:
        pass
