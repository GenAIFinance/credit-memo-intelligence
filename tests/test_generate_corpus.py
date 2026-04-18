"""
tests/test_generate_corpus.py

Unit tests for src/generation/generate_corpus.py.
Covers all five areas flagged by Codex review:
  1. Validation coercion
  2. Loop completion with partial batches
  3. Response shape parsing
  4. Resume behaviour
  5. Failure-mode termination
"""

import json
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ── Make src importable without install ───────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.generate_corpus import (
    MAX_CONSECUTIVE_EMPTY,
    OPTIONAL_NUMERIC,
    REQUIRED_FIELDS,
    REQUIRED_NUMERIC,
    _parse_response,
    _to_bool,
    _to_float,
    _to_int,
    count_existing,
    validate_doc,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _base_doc(**overrides) -> dict:
    """Return a minimal valid document dict, with optional field overrides."""
    doc = {
        "doc_id": str(uuid.uuid4()),
        "issuer": "Acme Holdings LLC",
        "borrower": None,
        "parent_issuer": "Acme PE Fund",
        "sector": "Healthcare Services",
        "rating_bucket": "B2/B",
        "date": "2023-06-15",
        "doc_type": "analyst_note",
        "event_type": "earnings_miss",
        "headline": "Acme misses Q2 EBITDA by 12% on volume shortfalls",
        "summary_text": "Quarterly EBITDA came in 12% below consensus.",
        "full_commentary": "Net leverage rose to 7.1x on weaker EBITDA. " * 3,
        "recommendations": "We recommend reducing exposure given covenant headroom below 10%.",
        "recommended_action": "reduce_exposure",
        "outcome_note": "Spread widened 75bps following the earnings release.",
        "exposure_change": -0.25,
        "spread_change": 75,
        "rating_change": -1,
        "watchlist_flag": True,
        "portfolio_action": "trim",
        "net_leverage": 7.1,
        "total_cap": 3200.0,
        "esg_score": 42.5,
        "collateral": "first_lien",
    }
    doc.update(overrides)
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# 1. Coercion helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestToFloat:
    def test_numeric_string(self):
        assert _to_float("6.8") == pytest.approx(6.8)

    def test_int_input(self):
        assert _to_float(7) == pytest.approx(7.0)

    def test_none_returns_default(self):
        assert _to_float(None) is None
        assert _to_float(None, default=0.0) == pytest.approx(0.0)

    def test_bad_string_returns_default(self):
        assert _to_float("N/A") is None
        assert _to_float("N/A", default=0.0) == pytest.approx(0.0)

    def test_empty_string_returns_default(self):
        assert _to_float("") is None


class TestToInt:
    def test_numeric_string(self):
        assert _to_int("75") == 75

    def test_float_string(self):
        assert _to_int("3.0") == 3

    def test_none_returns_default(self):
        assert _to_int(None) == 0

    def test_bad_string_returns_default(self):
        assert _to_int("N/A") == 0


class TestToBool:
    @pytest.mark.parametrize("value", [True, 1, "1", "true", "True", "TRUE", "yes", "y"])
    def test_truthy_values(self, value):
        assert _to_bool(value) is True

    @pytest.mark.parametrize("value", [False, 0, "0", "false", "False", "FALSE", "no", "n", None, ""])
    def test_falsy_values(self, value):
        assert _to_bool(value) is False

    def test_string_false_is_false(self):
        """Critical: 'false' string must not be coerced to True via bool()."""
        assert _to_bool("false") is False
        assert _to_bool("False") is False

    def test_unknown_value_defaults_false(self):
        assert _to_bool("maybe") is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. validate_doc
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateDoc:
    def test_valid_doc_passes(self):
        result = validate_doc(_base_doc())
        assert result is not None
        assert result["issuer"] == "Acme Holdings LLC"

    def test_missing_required_field_drops_doc(self):
        for field in REQUIRED_FIELDS:
            doc = _base_doc()
            doc.pop(field)
            assert validate_doc(doc) is None, f"Expected None when {field} is missing"

    def test_missing_required_numeric_drops_doc(self):
        doc = _base_doc()
        doc.pop("total_cap")
        assert validate_doc(doc) is None

    def test_malformed_required_numeric_drops_doc(self):
        doc = _base_doc(total_cap="N/A")
        assert validate_doc(doc) is None

    def test_malformed_optional_numeric_becomes_none(self):
        doc = _base_doc(esg_score="N/A", net_leverage="not-a-number")
        result = validate_doc(doc)
        assert result is not None
        assert result["esg_score"] is None
        assert result["net_leverage"] is None

    def test_watchlist_string_false_is_false(self):
        doc = _base_doc(watchlist_flag="false")
        result = validate_doc(doc)
        assert result is not None
        assert result["watchlist_flag"] is False

    def test_watchlist_string_true_is_true(self):
        doc = _base_doc(watchlist_flag="true")
        result = validate_doc(doc)
        assert result is not None
        assert result["watchlist_flag"] is True

    def test_invalid_recommended_action_gets_default(self):
        doc = _base_doc(recommended_action="do_nothing")
        result = validate_doc(doc)
        assert result is not None
        assert result["recommended_action"] == "maintain_position"

    def test_invalid_portfolio_action_gets_default(self):
        doc = _base_doc(portfolio_action="yolo")
        result = validate_doc(doc)
        assert result is not None
        assert result["portfolio_action"] == "monitor"

    def test_invalid_collateral_becomes_none(self):
        doc = _base_doc(collateral="second_mortgage")
        result = validate_doc(doc)
        assert result is not None
        assert result["collateral"] is None

    def test_missing_doc_id_gets_generated(self):
        doc = _base_doc()
        doc.pop("doc_id")
        result = validate_doc(doc)
        assert result is not None
        assert "doc_id" in result
        assert len(result["doc_id"]) == 36  # UUID format

    def test_non_dict_input_returns_none(self):
        assert validate_doc("not a dict") is None  # type: ignore
        assert validate_doc(None) is None  # type: ignore
        assert validate_doc([1, 2, 3]) is None  # type: ignore

    def test_spread_change_malformed_defaults_zero(self):
        doc = _base_doc(spread_change="wide")
        result = validate_doc(doc)
        assert result is not None
        assert result["spread_change"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Response shape parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestParseResponse:
    def test_root_list(self):
        payload = [{"a": 1}, {"b": 2}]
        result = _parse_response(json.dumps(payload), expected_count=2)
        assert result == payload

    def test_documents_key(self):
        payload = {"documents": [{"a": 1}]}
        result = _parse_response(json.dumps(payload), expected_count=1)
        assert result == payload["documents"]

    def test_unexpected_dict_raises(self):
        payload = {"items": [{"a": 1}]}  # not "documents"
        with pytest.raises(ValueError, match="Unexpected response shape"):
            _parse_response(json.dumps(payload), expected_count=1)

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_response("not json at all {", expected_count=1)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty document list"):
            _parse_response(json.dumps([]), expected_count=5)

    def test_documents_value_not_list_raises(self):
        payload = {"documents": "oops"}
        with pytest.raises(ValueError, match="expected list"):
            _parse_response(json.dumps(payload), expected_count=1)

    def test_truncated_json_raises(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_response('[{"issuer": "Acme"', expected_count=1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Resume support
# ─────────────────────────────────────────────────────────────────────────────

class TestCountExisting:
    def test_nonexistent_file_returns_zero(self, tmp_path):
        assert count_existing(tmp_path / "missing.jsonl") == 0

    def test_counts_nonempty_lines(self, tmp_path):
        f = tmp_path / "corpus.jsonl"
        f.write_text('{"a": 1}\n{"b": 2}\n\n{"c": 3}\n')
        assert count_existing(f) == 3

    def test_empty_file_returns_zero(self, tmp_path):
        f = tmp_path / "corpus.jsonl"
        f.write_text("")
        assert count_existing(f) == 0

    def test_handles_blank_lines(self, tmp_path):
        f = tmp_path / "corpus.jsonl"
        f.write_text("\n\n\n")
        assert count_existing(f) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Loop completion — while loop reaches target despite partial batches
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateLoop:
    """Test the while loop logic via generate() with mocked Azure calls."""

    def _make_doc(self) -> dict:
        return _base_doc(doc_id=str(uuid.uuid4()))

    def test_partial_batch_keeps_looping(self, tmp_path):
        """If a batch returns fewer valid docs than requested, loop continues."""
        from src.generation.generate_corpus import generate

        call_count = 0

        def fake_call_azure(client, prompt, batch_size):
            nonlocal call_count
            call_count += 1
            # First call returns only 3 docs (requested 5)
            # Subsequent calls return full batch
            if call_count == 1:
                return [self._make_doc() for _ in range(3)]
            return [self._make_doc() for _ in range(batch_size)]

        out_path = tmp_path / "corpus.jsonl"

        with patch("src.generation.generate_corpus.CFG", {
            "corpus": {
                "target_docs": 10,
                "batch_size": 5,
                "output_path": str(out_path),
            },
            "azure_openai": {
                "max_tokens": 100,
                "temperature": 0.8,
                "api_version": "2024-02-01",
                "chat_deployment": "gpt-4o",
            },
            "logging": {"level": "WARNING"},
        }), patch("src.generation.generate_corpus.get_llm_client", return_value=MagicMock()), \
           patch("src.generation.generate_corpus.call_azure", side_effect=fake_call_azure), \
           patch("src.generation.generate_corpus.build_prompt", return_value="mock prompt"), \
           patch("src.generation.generate_corpus.ROOT", tmp_path), \
           patch("src.generation.generate_corpus.time.sleep"):
            generate(target=10)

        written = count_existing(out_path)
        assert written >= 10, f"Expected >=10 docs, got {written}"
        # Should have needed more than 2 batches (10/5=2) because first was partial
        assert call_count > 2

    def test_consecutive_empty_batches_aborts(self, tmp_path):
        """Loop aborts after MAX_CONSECUTIVE_EMPTY batches with 0 valid docs."""
        from src.generation.generate_corpus import generate

        call_count = 0

        def fake_call_azure(client, prompt, batch_size):
            nonlocal call_count
            call_count += 1
            return [{"not_a_valid_doc": True}]  # will all fail validation

        out_path = tmp_path / "corpus.jsonl"

        with patch("src.generation.generate_corpus.CFG", {
            "corpus": {
                "target_docs": 20,
                "batch_size": 5,
                "output_path": str(out_path),
            },
            "azure_openai": {
                "max_tokens": 100,
                "temperature": 0.8,
                "api_version": "2024-02-01",
                "chat_deployment": "gpt-4o",
            },
            "logging": {"level": "WARNING"},
        }), patch("src.generation.generate_corpus.get_llm_client", return_value=MagicMock()), \
           patch("src.generation.generate_corpus.call_azure", side_effect=fake_call_azure), \
           patch("src.generation.generate_corpus.build_prompt", return_value="mock prompt"), \
           patch("src.generation.generate_corpus.ROOT", tmp_path), \
           patch("src.generation.generate_corpus.time.sleep"):
            generate(target=20)

        # Should have stopped after MAX_CONSECUTIVE_EMPTY attempts
        assert call_count == MAX_CONSECUTIVE_EMPTY

    def test_invalid_target_raises(self, tmp_path):
        from src.generation.generate_corpus import generate
        with patch("src.generation.generate_corpus.CFG", {
            "corpus": {"target_docs": -1, "batch_size": 5, "output_path": str(tmp_path / "x.jsonl")},
            "azure_openai": {"max_tokens": 100, "temperature": 0.8, "api_version": "2024", "chat_deployment": "gpt-4o"},
            "logging": {"level": "WARNING"},
        }), patch("src.generation.generate_corpus.ROOT", tmp_path):
            with pytest.raises(ValueError, match="positive integer"):
                generate(target=-1)

    def test_explicit_zero_target_raises(self, tmp_path):
        """target=0 must raise immediately — not silently fall back to config default."""
        from src.generation.generate_corpus import generate
        with patch("src.generation.generate_corpus.CFG", {
            "corpus": {"target_docs": 3000, "batch_size": 5, "output_path": str(tmp_path / "x.jsonl")},
            "azure_openai": {"max_tokens": 100, "temperature": 0.8, "api_version": "2024", "chat_deployment": "gpt-4o"},
            "logging": {"level": "WARNING"},
        }), patch("src.generation.generate_corpus.ROOT", tmp_path):
            with pytest.raises(ValueError, match="positive integer"):
                generate(target=0)

    def test_resume_noop_does_not_require_azure_client(self, tmp_path):
        """If corpus already meets target, no Azure client should be created."""
        from src.generation.generate_corpus import generate

        # Pre-populate output with enough docs to satisfy target
        out_path = tmp_path / "corpus.jsonl"
        for _ in range(5):
            out_path.open("a").write('{"doc_id": "x"}\n')

        client_calls = []

        with patch("src.generation.generate_corpus.CFG", {
            "corpus": {"target_docs": 5, "batch_size": 5, "output_path": str(out_path)},
            "azure_openai": {"max_tokens": 100, "temperature": 0.8, "api_version": "2024", "chat_deployment": "gpt-4o"},
            "logging": {"level": "WARNING"},
        }), patch("src.generation.generate_corpus.ROOT", tmp_path), \
           patch("src.generation.generate_corpus.get_llm_client", side_effect=lambda: client_calls.append(1) or MagicMock()):
            generate(target=5)

        assert len(client_calls) == 0, "build_client() must not be called on a no-op resume"

    def test_batch_overshoot_capped_at_target(self, tmp_path):
        """If model returns more valid docs than needed, written count must not exceed target."""
        from src.generation.generate_corpus import generate

        def fake_call_azure(client, prompt, batch_size):
            # Return double the requested batch — model over-returning
            return [self._make_doc() for _ in range(batch_size * 2)]

        out_path = tmp_path / "corpus.jsonl"

        with patch("src.generation.generate_corpus.CFG", {
            "corpus": {"target_docs": 7, "batch_size": 5, "output_path": str(out_path)},
            "azure_openai": {"max_tokens": 100, "temperature": 0.8, "api_version": "2024", "chat_deployment": "gpt-4o"},
            "logging": {"level": "WARNING"},
        }), patch("src.generation.generate_corpus.get_llm_client", return_value=MagicMock()), \
           patch("src.generation.generate_corpus.call_azure", side_effect=fake_call_azure), \
           patch("src.generation.generate_corpus.build_prompt", return_value="mock prompt"), \
           patch("src.generation.generate_corpus.ROOT", tmp_path), \
           patch("src.generation.generate_corpus.time.sleep"):
            generate(target=7)

        written = count_existing(out_path)
        assert written == 7, f"Expected exactly 7 docs, got {written}"
