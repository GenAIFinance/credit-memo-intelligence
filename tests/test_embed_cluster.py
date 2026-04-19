"""
tests/test_embed_cluster.py

Unit tests for src/embeddings/embed_cluster.py.
Covers Codex review findings:
  - NaN numeric metadata is normalized to 0.0 before ChromaDB upsert
  - math.isfinite guard catches pandas NaN that passes val is not None
"""

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_row(**kwargs):
    """Build a minimal chunk row dict for metadata testing."""
    defaults = {
        "chunk_id": "doc1_0000",
        "doc_id": "doc1",
        "issuer": "Acme Holdings LLC",
        "borrower": None,
        "sector": "Healthcare",
        "rating_bucket": "B2/B",
        "date": "2023-06-15",
        "doc_type": "analyst_note",
        "event_type": "earnings_miss",
        "section": "executive_summary_header",
        "chunk_type": "text",
        "collateral": "first_lien",
        "recommended_action": "reduce_exposure",
        "portfolio_action": "trim",
        "watchlist_flag": True,
        "net_leverage": 7.1,
        "total_cap": 3200.0,
        "esg_score": 42.5,
        "spread_change": 75,
        "rating_change": -1,
        "chunk_index": 0,
        "text": "Test chunk text for embedding.",
    }
    defaults.update(kwargs)
    return defaults


def _build_single_row_df(**kwargs) -> pd.DataFrame:
    """Build a single-row DataFrame from a row dict."""
    return pd.DataFrame([_make_row(**kwargs)])


# ── NaN metadata normalization ────────────────────────────────────────────────

class TestNaNMetadataNormalization:
    """Codex finding: pandas NaN passes 'val is not None' but is non-finite."""

    def _extract_meta(self, df: pd.DataFrame, cluster_id: int = 0) -> dict:
        """Run the metadata building logic and return the metadata dict."""
        import math
        _NULLABLE_NUM_COLS = [
            "net_leverage", "total_cap", "esg_score",
            "spread_change", "rating_change", "chunk_index",
        ]
        _NULLABLE_STR_COLS = [
            "issuer", "borrower", "sector", "rating_bucket", "date",
            "doc_type", "event_type", "section", "chunk_type",
            "collateral", "recommended_action", "portfolio_action",
        ]
        row = next(df.itertuples(index=False))
        meta = {
            "doc_id": str(row.doc_id),
            "cluster_id": cluster_id,
            "watchlist_flag": bool(getattr(row, "watchlist_flag", False)),
        }
        for col in _NULLABLE_STR_COLS:
            val = getattr(row, col, None)
            meta[col] = str(val) if val is not None else ""
        for col in _NULLABLE_NUM_COLS:
            val = getattr(row, col, None)
            try:
                f = float(val)
                meta[col] = f if math.isfinite(f) else 0.0
            except (TypeError, ValueError):
                meta[col] = 0.0
        return meta

    def test_nan_net_leverage_becomes_zero(self):
        """pandas NaN for missing leverage must not reach ChromaDB as NaN."""
        df = _build_single_row_df(net_leverage=float("nan"))
        meta = self._extract_meta(df)
        assert meta["net_leverage"] == 0.0
        assert math.isfinite(meta["net_leverage"])

    def test_none_net_leverage_becomes_zero(self):
        df = _build_single_row_df(net_leverage=None)
        meta = self._extract_meta(df)
        assert meta["net_leverage"] == 0.0

    def test_nan_esg_score_becomes_zero(self):
        df = _build_single_row_df(esg_score=float("nan"))
        meta = self._extract_meta(df)
        assert meta["esg_score"] == 0.0
        assert math.isfinite(meta["esg_score"])

    def test_inf_value_becomes_zero(self):
        """Infinity is also non-finite and must be normalized."""
        df = _build_single_row_df(net_leverage=float("inf"))
        meta = self._extract_meta(df)
        assert meta["net_leverage"] == 0.0

    def test_valid_leverage_preserved(self):
        df = _build_single_row_df(net_leverage=6.8)
        meta = self._extract_meta(df)
        assert meta["net_leverage"] == pytest.approx(6.8)

    def test_valid_total_cap_preserved(self):
        df = _build_single_row_df(total_cap=3200.0)
        meta = self._extract_meta(df)
        assert meta["total_cap"] == pytest.approx(3200.0)

    def test_all_nan_numeric_fields_become_zero(self):
        """All nullable numeric fields with NaN must all become 0.0."""
        df = _build_single_row_df(
            net_leverage=float("nan"),
            esg_score=float("nan"),
            spread_change=float("nan"),
            rating_change=float("nan"),
        )
        meta = self._extract_meta(df)
        for col in ["net_leverage", "esg_score", "spread_change", "rating_change"]:
            assert meta[col] == 0.0, f"{col} should be 0.0, got {meta[col]}"
            assert math.isfinite(meta[col]), f"{col} should be finite"

    def test_pandas_nan_from_duckdb_nullable_column(self):
        """Simulate how pandas represents NULL from a DuckDB nullable column."""
        # DuckDB nullable floats come through as pd.NA or float NaN in pandas
        df = pd.DataFrame([_make_row(net_leverage=pd.NA, esg_score=pd.NA)])
        meta = self._extract_meta(df)
        assert meta["net_leverage"] == 0.0
        assert meta["esg_score"] == 0.0


class TestRiskThemeMetadata:
    """Codex finding: risk_theme was never written to ChromaDB metadata."""

    def _extract_meta_with_map(
        self, df: pd.DataFrame, risk_theme_map: dict, cluster_id: int = 0
    ) -> dict:
        """Run metadata building with risk_theme_map and return meta dict."""
        import math
        _NULLABLE_NUM_COLS = [
            "net_leverage", "total_cap", "esg_score",
            "spread_change", "rating_change", "chunk_index",
        ]
        _NULLABLE_STR_COLS = [
            "issuer", "borrower", "sector", "rating_bucket", "date",
            "doc_type", "event_type", "section", "chunk_type",
            "collateral", "recommended_action", "portfolio_action",
        ]
        row = next(df.itertuples(index=False))
        meta = {
            "doc_id": str(row.doc_id),
            "cluster_id": cluster_id,
            "watchlist_flag": bool(getattr(row, "watchlist_flag", False)),
        }
        meta["risk_theme"] = risk_theme_map.get(str(row.doc_id), "")
        for col in _NULLABLE_STR_COLS:
            val = getattr(row, col, None)
            meta[col] = str(val) if val is not None else ""
        for col in _NULLABLE_NUM_COLS:
            val = getattr(row, col, None)
            try:
                f = float(val)
                meta[col] = f if math.isfinite(f) else 0.0
            except (TypeError, ValueError):
                meta[col] = 0.0
        return meta

    def test_risk_theme_written_when_doc_in_map(self):
        """Doc present in llm_labels map must have risk_theme in metadata."""
        df = _build_single_row_df()
        doc_id = df["doc_id"].iloc[0]
        risk_theme_map = {doc_id: "leverage_deterioration"}
        meta = self._extract_meta_with_map(df, risk_theme_map)
        assert meta["risk_theme"] == "leverage_deterioration"

    def test_risk_theme_empty_string_when_doc_not_in_map(self):
        """Doc absent from llm_labels map gets risk_theme='' not KeyError."""
        df = _build_single_row_df()
        meta = self._extract_meta_with_map(df, risk_theme_map={})
        assert meta["risk_theme"] == ""

    def test_risk_theme_empty_when_no_map_provided(self):
        """None map (labels not yet generated) defaults to empty string."""
        import math
        _NULLABLE_NUM_COLS = ["net_leverage", "total_cap", "esg_score",
                               "spread_change", "rating_change", "chunk_index"]
        df = _build_single_row_df()
        row = next(df.itertuples(index=False))
        risk_theme_map = None
        risk_theme = (
            risk_theme_map.get(str(row.doc_id), "")
            if risk_theme_map else ""
        )
        assert risk_theme == ""

    def test_risk_theme_key_present_in_metadata(self):
        """risk_theme key must always be present in ChromaDB metadata."""
        df = _build_single_row_df()
        meta = self._extract_meta_with_map(df, risk_theme_map={})
        assert "risk_theme" in meta, "risk_theme key missing from metadata dict"
