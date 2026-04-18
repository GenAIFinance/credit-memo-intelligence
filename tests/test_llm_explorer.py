"""
tests/test_llm_explorer.py

Unit tests for src/exploration/llm_explorer.py.
Covers Codex review finding:
  - ambiguous flag must not treat string "false" as True
  - _to_bool handles all boolean string variants correctly
  - _parse_label correctly coerces ambiguous and confidence fields
  - _stratified_sample respects n and covers all strata
"""

import json
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_label_json(**overrides) -> str:
    """Return a valid label JSON string with optional field overrides."""
    label = {
        "risk_theme": "leverage_deterioration",
        "action": "reduce_exposure",
        "outcome": "spread_widening",
        "trigger": "Q3 EBITDA missed by 18%.",
        "risk_signal": "Covenant headroom below 10%.",
        "analyst_stance": "Reduce exposure given near-term covenant risk.",
        "forward_view": "Spread likely to widen if leverage does not improve.",
        "confidence": 0.85,
        "ambiguous": False,
        "new_label_proposed": None,
    }
    label.update(overrides)
    return json.dumps(label)


def _base_doc(**overrides) -> dict:
    doc = {
        "doc_id": str(uuid.uuid4()),
        "issuer": "Acme Holdings LLC",
        "sector": "Healthcare Services",
        "rating_bucket": "B2/B",
        "doc_type": "analyst_note",
        "event_type": "earnings_miss",
        "headline": "Acme misses Q2 EBITDA.",
        "summary_text": "EBITDA came in 12% below consensus.",
        "full_commentary": "Net leverage rose to 7.1x. Covenant headroom tightened.",
        "recommendations": "Reduce exposure on covenant risk.",
        "recommended_action": "reduce_exposure",
        "net_leverage": 7.1,
        "collateral": "first_lien",
    }
    doc.update(overrides)
    return doc


# ── _to_bool — the Codex finding ─────────────────────────────────────────────

class TestToBool:
    """Codex finding: bool('false') returns True — _to_bool must handle this."""

    def test_string_false_returns_false(self):
        from src.exploration.llm_explorer import _to_bool
        assert _to_bool("false") is False

    def test_string_False_returns_false(self):
        from src.exploration.llm_explorer import _to_bool
        assert _to_bool("False") is False

    def test_string_true_returns_true(self):
        from src.exploration.llm_explorer import _to_bool
        assert _to_bool("true") is True

    def test_bool_false_returns_false(self):
        from src.exploration.llm_explorer import _to_bool
        assert _to_bool(False) is False

    def test_bool_true_returns_true(self):
        from src.exploration.llm_explorer import _to_bool
        assert _to_bool(True) is True

    def test_none_returns_false(self):
        from src.exploration.llm_explorer import _to_bool
        assert _to_bool(None) is False

    def test_unknown_value_returns_false(self):
        from src.exploration.llm_explorer import _to_bool
        assert _to_bool("maybe") is False


# ── _parse_label ──────────────────────────────────────────────────────────────

class TestParseLabel:
    def test_string_false_ambiguous_parsed_correctly(self):
        """Core Codex finding: 'false' string must not mark doc as ambiguous."""
        from src.exploration.llm_explorer import _parse_label
        raw = _valid_label_json(ambiguous="false")
        result = _parse_label(raw, doc_id="test-001")
        assert result is not None
        assert result["ambiguous"] is False

    def test_string_true_ambiguous_parsed_correctly(self):
        from src.exploration.llm_explorer import _parse_label
        raw = _valid_label_json(ambiguous="true")
        result = _parse_label(raw, doc_id="test-002")
        assert result is not None
        assert result["ambiguous"] is True

    def test_bool_false_ambiguous_parsed_correctly(self):
        from src.exploration.llm_explorer import _parse_label
        raw = _valid_label_json(ambiguous=False)
        result = _parse_label(raw, doc_id="test-003")
        assert result is not None
        assert result["ambiguous"] is False

    def test_confidence_coerced_to_float(self):
        from src.exploration.llm_explorer import _parse_label
        raw = _valid_label_json(confidence="0.72")
        result = _parse_label(raw, doc_id="test-004")
        assert result is not None
        assert result["confidence"] == pytest.approx(0.72)

    def test_bad_confidence_defaults_zero(self):
        from src.exploration.llm_explorer import _parse_label
        raw = _valid_label_json(confidence="high")
        result = _parse_label(raw, doc_id="test-005")
        assert result is not None
        assert result["confidence"] == 0.0

    def test_missing_required_field_returns_none(self):
        from src.exploration.llm_explorer import _parse_label
        label = json.loads(_valid_label_json())
        label.pop("risk_theme")
        result = _parse_label(json.dumps(label), doc_id="test-006")
        assert result is None

    def test_invalid_json_returns_none(self):
        from src.exploration.llm_explorer import _parse_label
        result = _parse_label("not json {", doc_id="test-007")
        assert result is None

    def test_doc_id_attached_to_label(self):
        from src.exploration.llm_explorer import _parse_label
        raw = _valid_label_json()
        result = _parse_label(raw, doc_id="my-doc-id")
        assert result is not None
        assert result["doc_id"] == "my-doc-id"


# ── _stratified_sample ────────────────────────────────────────────────────────

class TestStratifiedSample:
    def _make_docs(self, n: int, doc_type: str) -> list[dict]:
        return [_base_doc(doc_id=str(uuid.uuid4()), doc_type=doc_type) for _ in range(n)]

    def test_sample_size_respected(self):
        from src.exploration.llm_explorer import _stratified_sample
        docs = (
            self._make_docs(50, "analyst_note") +
            self._make_docs(50, "watchlist_note")
        )
        sample = _stratified_sample(docs, n=20)
        assert len(sample) == 20

    def test_all_strata_represented(self):
        from src.exploration.llm_explorer import _stratified_sample
        docs = (
            self._make_docs(40, "analyst_note") +
            self._make_docs(40, "watchlist_note") +
            self._make_docs(20, "rating_action")
        )
        sample = _stratified_sample(docs, n=30)
        types = {d["doc_type"] for d in sample}
        assert "analyst_note" in types
        assert "watchlist_note" in types
        assert "rating_action" in types

    def test_sample_larger_than_corpus_returns_corpus(self):
        from src.exploration.llm_explorer import _stratified_sample
        docs = self._make_docs(5, "analyst_note")
        sample = _stratified_sample(docs, n=100)
        assert len(sample) == 5

    def test_no_duplicates_in_sample(self):
        from src.exploration.llm_explorer import _stratified_sample
        docs = self._make_docs(100, "analyst_note")
        sample = _stratified_sample(docs, n=30)
        ids = [d["doc_id"] for d in sample]
        assert len(ids) == len(set(ids))
