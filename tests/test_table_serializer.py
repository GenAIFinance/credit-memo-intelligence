"""
tests/test_table_serializer.py

Unit tests for src/preprocessing/table_serializer.py.
Covers Codex review findings:
  - serialize({}) returns empty string
  - API exception path returns template fallback when enabled
  - API exception path re-raises when fallback disabled
  - serialize_cap_table includes formatted leverage/cap/spread fields
  - max_retries is respected (not hardcoded)
  - None content from API raises, triggering fallback
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_serializer(fallback: bool = True, max_retries: int = 1):
    """Build a TableSerializer with mocked LLM client."""
    from src.preprocessing.table_serializer import TableSerializer
    s = TableSerializer(max_retries=max_retries, fallback_to_template=fallback)
    s._client = MagicMock()
    return s


def _mock_response(content: str | None) -> MagicMock:
    """Build a mock API response object."""
    resp = MagicMock()
    resp.choices[0].message.content = content
    return resp


# ── Empty input ───────────────────────────────────────────────────────────────

class TestSerializeEmptyInput:
    def test_empty_dict_returns_empty_string(self):
        from src.preprocessing.table_serializer import TableSerializer
        s = TableSerializer()
        with patch("src.preprocessing.table_serializer.get_llm_client", return_value=MagicMock()):
            result = s.serialize({})
        assert result == ""

    def test_none_values_are_skipped_in_format(self):
        from src.preprocessing.table_serializer import _format_table_data
        result = _format_table_data({"A": "1", "B": None, "C": "3"})
        assert "B" not in result
        assert "A: 1" in result
        assert "C: 3" in result


# ── Fallback behavior ─────────────────────────────────────────────────────────

class TestSerializeFallback:
    def test_api_failure_returns_template_when_fallback_enabled(self):
        s = _make_serializer(fallback=True, max_retries=1)
        s._client.chat.completions.create.side_effect = ConnectionError("timeout")

        with patch("src.preprocessing.table_serializer.get_model_name", return_value="gpt-4o"), \
             patch("src.preprocessing.table_serializer.get_llm_client", return_value=s._client):
            result = s.serialize({"Net Leverage": "6.8x"}, "cap table")

        assert result != ""
        assert "6.8x" in result  # template fallback includes the values

    def test_api_failure_reraises_when_fallback_disabled(self):
        s = _make_serializer(fallback=False, max_retries=1)
        s._client.chat.completions.create.side_effect = ConnectionError("timeout")

        with patch("src.preprocessing.table_serializer.get_model_name", return_value="gpt-4o"), \
             patch("src.preprocessing.table_serializer.get_llm_client", return_value=s._client):
            with pytest.raises(Exception):
                s.serialize({"Net Leverage": "6.8x"}, "cap table")

    def test_none_content_triggers_fallback(self):
        """API returning None content should fall back gracefully."""
        s = _make_serializer(fallback=True, max_retries=1)
        s._client.chat.completions.create.return_value = _mock_response(None)

        with patch("src.preprocessing.table_serializer.get_model_name", return_value="gpt-4o"), \
             patch("src.preprocessing.table_serializer.get_llm_client", return_value=s._client):
            result = s.serialize({"Leverage": "7.1x"}, "cap table")

        assert result != ""  # fallback string, not crash

    def test_successful_api_call_returns_prose(self):
        s = _make_serializer(fallback=True, max_retries=1)
        s._client.chat.completions.create.return_value = _mock_response(
            "Net leverage of 6.8x on total capitalization of $3,200mm."
        )

        with patch("src.preprocessing.table_serializer.get_model_name", return_value="gpt-4o"), \
             patch("src.preprocessing.table_serializer.get_llm_client", return_value=s._client):
            result = s.serialize({"Net Leverage": "6.8x"}, "cap table")

        assert "6.8x" in result


# ── max_retries is respected ──────────────────────────────────────────────────

class TestMaxRetries:
    def test_max_retries_one_calls_api_once(self):
        """With max_retries=1, a transient failure should not retry."""
        s = _make_serializer(fallback=True, max_retries=1)
        s._client.chat.completions.create.side_effect = ConnectionError("fail")

        with patch("src.preprocessing.table_serializer.get_model_name", return_value="gpt-4o"), \
             patch("src.preprocessing.table_serializer.get_llm_client", return_value=s._client):
            s.serialize({"A": "1"}, "test")

        assert s._client.chat.completions.create.call_count == 1

    def test_max_retries_three_retries_on_transient(self):
        """With max_retries=3, transient failures should be retried up to 3 times."""
        s = _make_serializer(fallback=True, max_retries=3)
        s._client.chat.completions.create.side_effect = ConnectionError("fail")

        with patch("src.preprocessing.table_serializer.get_model_name", return_value="gpt-4o"), \
             patch("src.preprocessing.table_serializer.get_llm_client", return_value=s._client):
            s.serialize({"A": "1"}, "test")

        assert s._client.chat.completions.create.call_count == 3


# ── serialize_cap_table ───────────────────────────────────────────────────────

class TestSerializeCapTable:
    def _base_doc(self, **overrides) -> dict:
        doc = {
            "net_leverage": 6.8,
            "total_cap": 3200.0,
            "collateral": "first_lien",
            "rating_bucket": "B2/B",
            "spread_change": 75,
            "esg_score": 42.5,
        }
        doc.update(overrides)
        return doc

    def test_leverage_formatted_as_x(self):
        from src.preprocessing.table_serializer import TableSerializer
        s = TableSerializer()
        table_data_captured = {}

        def fake_serialize(table_data, table_context=""):
            table_data_captured.update(table_data)
            return "prose"

        s.serialize = fake_serialize
        s.serialize_cap_table(self._base_doc())
        assert "Net Leverage" in table_data_captured
        assert "6.8x" in table_data_captured["Net Leverage"]

    def test_total_cap_formatted_with_mm(self):
        from src.preprocessing.table_serializer import TableSerializer
        s = TableSerializer()
        table_data_captured = {}

        def fake_serialize(table_data, table_context=""):
            table_data_captured.update(table_data)
            return "prose"

        s.serialize = fake_serialize
        s.serialize_cap_table(self._base_doc())
        assert "Total Capitalization" in table_data_captured
        assert "$3,200mm" in table_data_captured["Total Capitalization"]

    def test_spread_widening_labeled_correctly(self):
        from src.preprocessing.table_serializer import TableSerializer
        s = TableSerializer()
        table_data_captured = {}

        def fake_serialize(table_data, table_context=""):
            table_data_captured.update(table_data)
            return "prose"

        s.serialize = fake_serialize
        s.serialize_cap_table(self._base_doc(spread_change=50))
        assert "widened" in table_data_captured.get("Spread Movement", "")

    def test_spread_tightening_labeled_correctly(self):
        from src.preprocessing.table_serializer import TableSerializer
        s = TableSerializer()
        table_data_captured = {}

        def fake_serialize(table_data, table_context=""):
            table_data_captured.update(table_data)
            return "prose"

        s.serialize = fake_serialize
        s.serialize_cap_table(self._base_doc(spread_change=-30))
        assert "tightened" in table_data_captured.get("Spread Movement", "")

    def test_zero_spread_change_excluded(self):
        from src.preprocessing.table_serializer import TableSerializer
        s = TableSerializer()
        table_data_captured = {}

        def fake_serialize(table_data, table_context=""):
            table_data_captured.update(table_data)
            return "prose"

        s.serialize = fake_serialize
        s.serialize_cap_table(self._base_doc(spread_change=0))
        assert "Spread Movement" not in table_data_captured

    def test_null_fields_excluded_from_table(self):
        from src.preprocessing.table_serializer import TableSerializer
        s = TableSerializer()
        table_data_captured = {}

        def fake_serialize(table_data, table_context=""):
            table_data_captured.update(table_data)
            return "prose"

        s.serialize = fake_serialize
        s.serialize_cap_table(self._base_doc(esg_score=None, net_leverage=None))
        assert "ESG Score" not in table_data_captured
        assert "Net Leverage" not in table_data_captured
