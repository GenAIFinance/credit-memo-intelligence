"""
tests/test_normalize.py

Unit tests for src/preprocessing/normalize.py.
Covers Codex review findings:
  - section extraction returns expected section names
  - chunk IDs are deterministic and unique per doc
  - malformed JSONL lines are skipped without aborting pipeline
"""

import json
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_doc(**overrides) -> dict:
    """Return a minimal valid corpus document."""
    doc = {
        "doc_id": str(uuid.uuid4()),
        "issuer": "Acme Holdings LLC",
        "borrower": None,
        "sector": "Healthcare Services",
        "rating_bucket": "B2/B",
        "date": "2023-06-15",
        "doc_type": "analyst_note",
        "event_type": "earnings_miss",
        "headline": "Acme misses Q2 EBITDA by 12% on volume shortfalls.",
        "summary_text": "Quarterly EBITDA came in 12% below consensus driven by volume shortfalls across all three business segments.",
        "full_commentary": (
            "Net leverage rose to 7.1x on weaker EBITDA. "
            "Interest coverage declined to 2.1x. "
            "Liquidity runway shortened to 8 months. "
            "Covenant headroom tightened to below 10%. "
            "Management expects recovery in H2 but visibility is limited. "
            "We remain cautious on the credit."
        ),
        "recommendations": (
            "We recommend reducing exposure. Covenant headroom below 10% "
            "creates near-term risk of technical default. Exit if leverage "
            "does not improve by next quarter."
        ),
        "recommended_action": "reduce_exposure",
        "outcome_note": "Spread widened 75bps following the earnings release.",
        "net_leverage": 7.1,
        "total_cap": 3200.0,
        "esg_score": 42.5,
        "collateral": "first_lien",
        "portfolio_action": "trim",
        "watchlist_flag": True,
        "spread_change": 75,
        "rating_change": -1,
        "exposure_change": -0.25,
    }
    doc.update(overrides)
    return doc


def _make_serializer_mock() -> MagicMock:
    """Return a mock TableSerializer that returns deterministic prose."""
    mock = MagicMock()
    mock.serialize_cap_table.return_value = (
        "Net leverage of 7.1x on total capitalization of $3,200mm. "
        "Transaction secured on a first lien basis."
    )
    return mock


# ── Section extraction ────────────────────────────────────────────────────────

class TestExtractSections:
    def test_expected_sections_present(self):
        from src.preprocessing.normalize import _extract_sections
        doc = _base_doc()
        mock_ser = _make_serializer_mock()
        sections = _extract_sections(doc, mock_ser)
        names = {s["section"] for s in sections}

        expected = {
            "executive_summary_header",
            "executive_summary_reco",
            "executive_summary_cap_table",
            "company_description",
            "transaction_overview",
        }
        assert expected.issubset(names), (
            f"Missing sections: {expected - names}"
        )

    def test_cap_table_chunk_type_is_table_serialized(self):
        from src.preprocessing.normalize import _extract_sections
        doc = _base_doc()
        mock_ser = _make_serializer_mock()
        sections = _extract_sections(doc, mock_ser)
        cap = next(
            (s for s in sections if s["section"] == "executive_summary_cap_table"),
            None,
        )
        assert cap is not None
        assert cap["chunk_type"] == "table_serialized"

    def test_text_sections_have_text_chunk_type(self):
        from src.preprocessing.normalize import _extract_sections
        doc = _base_doc()
        mock_ser = _make_serializer_mock()
        sections = _extract_sections(doc, mock_ser)
        text_sections = [s for s in sections if s["section"] != "executive_summary_cap_table"]
        for s in text_sections:
            assert s["chunk_type"] == "text", (
                f"Section '{s['section']}' should be 'text', got '{s['chunk_type']}'"
            )

    def test_short_text_below_min_length_excluded(self):
        from src.preprocessing.normalize import _extract_sections
        # Summary text too short to produce a section
        doc = _base_doc(summary_text="Short.")
        mock_ser = _make_serializer_mock()
        sections = _extract_sections(doc, mock_ser)
        names = {s["section"] for s in sections}
        assert "company_description" not in names

    def test_header_contains_issuer_name(self):
        from src.preprocessing.normalize import _section_executive_header
        doc = _base_doc()
        text = _section_executive_header(doc)
        assert "Acme Holdings LLC" in text

    def test_header_contains_leverage(self):
        from src.preprocessing.normalize import _section_executive_header
        doc = _base_doc(net_leverage=7.1)
        text = _section_executive_header(doc)
        assert "7.1x" in text

    def test_reco_contains_recommended_action(self):
        from src.preprocessing.normalize import _section_executive_reco
        doc = _base_doc(recommended_action="reduce_exposure")
        text = _section_executive_reco(doc)
        assert "reduce exposure" in text.lower()

    def test_watchlist_flag_appears_in_reco(self):
        from src.preprocessing.normalize import _section_executive_reco
        doc = _base_doc(watchlist_flag=True)
        text = _section_executive_reco(doc)
        assert "watchlist" in text.lower()


# ── Chunk ID uniqueness and determinism ───────────────────────────────────────

class TestChunkIds:
    def test_chunk_ids_are_unique_within_doc(self):
        from src.preprocessing.normalize import (
            _build_chunks, _extract_sections, _build_splitter, _memo_metadata
        )
        doc = _base_doc()
        mock_ser = _make_serializer_mock()
        splitter = _build_splitter()
        sections = _extract_sections(doc, mock_ser)
        meta = _memo_metadata(doc)
        chunks = _build_chunks(doc, sections, splitter, meta)

        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_chunk_ids_start_with_doc_id(self):
        from src.preprocessing.normalize import (
            _build_chunks, _extract_sections, _build_splitter, _memo_metadata
        )
        doc = _base_doc()
        mock_ser = _make_serializer_mock()
        splitter = _build_splitter()
        sections = _extract_sections(doc, mock_ser)
        meta = _memo_metadata(doc)
        chunks = _build_chunks(doc, sections, splitter, meta)

        for chunk in chunks:
            assert chunk["chunk_id"].startswith(doc["doc_id"]), (
                f"chunk_id '{chunk['chunk_id']}' does not start with doc_id"
            )

    def test_chunk_ids_are_deterministic(self):
        """Same doc produces same chunk IDs on repeated calls."""
        from src.preprocessing.normalize import (
            _build_chunks, _extract_sections, _build_splitter, _memo_metadata
        )
        doc = _base_doc()
        mock_ser = _make_serializer_mock()
        splitter = _build_splitter()

        sections1 = _extract_sections(doc, mock_ser)
        chunks1 = _build_chunks(doc, sections1, splitter, _memo_metadata(doc))

        sections2 = _extract_sections(doc, mock_ser)
        chunks2 = _build_chunks(doc, sections2, splitter, _memo_metadata(doc))

        ids1 = [c["chunk_id"] for c in chunks1]
        ids2 = [c["chunk_id"] for c in chunks2]
        assert ids1 == ids2

    def test_memo_metadata_on_every_chunk(self):
        """Every chunk carries the memo-level metrics for retrieval filtering."""
        from src.preprocessing.normalize import (
            _build_chunks, _extract_sections, _build_splitter, _memo_metadata
        )
        doc = _base_doc(net_leverage=7.1, collateral="first_lien")
        mock_ser = _make_serializer_mock()
        splitter = _build_splitter()
        sections = _extract_sections(doc, mock_ser)
        meta = _memo_metadata(doc)
        chunks = _build_chunks(doc, sections, splitter, meta)

        for chunk in chunks:
            assert chunk.get("net_leverage") == 7.1
            assert chunk.get("collateral") == "first_lien"
            assert chunk.get("issuer") == "Acme Holdings LLC"


# ── Malformed JSONL handling ──────────────────────────────────────────────────

class TestReadCorpus:
    def test_malformed_lines_skipped(self, tmp_path):
        from src.preprocessing.normalize import _read_corpus
        f = tmp_path / "corpus.jsonl"
        good = json.dumps(_base_doc())
        f.write_text(
            f"{good}\n"
            "this is not json\n"
            f"{good}\n"
            "{incomplete json\n"
            f"{good}\n"
        )
        docs = list(_read_corpus(f))
        assert len(docs) == 3

    def test_blank_lines_skipped(self, tmp_path):
        from src.preprocessing.normalize import _read_corpus
        f = tmp_path / "corpus.jsonl"
        good = json.dumps(_base_doc())
        f.write_text(f"\n\n{good}\n\n{good}\n\n")
        docs = list(_read_corpus(f))
        assert len(docs) == 2

    def test_missing_file_raises(self, tmp_path):
        from src.preprocessing.normalize import _read_corpus
        with pytest.raises(FileNotFoundError):
            list(_read_corpus(tmp_path / "missing.jsonl"))

    def test_limit_respected(self, tmp_path):
        from src.preprocessing.normalize import _read_corpus
        f = tmp_path / "corpus.jsonl"
        lines = "\n".join(json.dumps(_base_doc()) for _ in range(10))
        f.write_text(lines)
        docs = list(_read_corpus(f, limit=3))
        assert len(docs) == 3
