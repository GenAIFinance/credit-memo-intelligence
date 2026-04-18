"""
src/preprocessing/normalize.py

Phase 2: Section-Aware Preprocessing + Chunking
------------------------------------------------
Reads the raw JSONL corpus from Phase 1 and produces a structured chunk
table in DuckDB. Each document is mapped to named sections, and each
section is split into chunks using LangChain's RecursiveCharacterTextSplitter.

This module implements two explicit LangChain contributions cited in the
job description:
  - Chunking module: RecursiveCharacterTextSplitter with section-aware
    metadata tagging
  - Table serialization: TableSerializer called as a pre-chunking
    transformation before embedding

Section mapping (MVP — 5 sections, 7 chunk types):
  executive_summary_header      text     issuer + ESG metadata as prose
  executive_summary_reco        text     recommendation + action
  executive_summary_cap_table   table    net_leverage, total_cap → serialized
  company_description           text     summary_text
  transaction_overview          text     headline + full_commentary
  merits_and_concerns           text     full_commentary split into 2 halves
  rep_risk_esg                  text     esg_score + outcome_note

All memo-level metrics are attached to every chunk as metadata so the
retrieval layer can filter any chunk by leverage, collateral, ESG, etc.

Usage:
    python -m src.preprocessing.normalize
    python -m src.preprocessing.normalize --limit 100   # quick test
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Generator

import duckdb
import pandas as pd
import yaml
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.preprocessing.table_serializer import TableSerializer

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

logging.basicConfig(
    level=getattr(logging, CFG["logging"]["level"]),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = CFG["preprocessing"]["chunk_size"]
CHUNK_OVERLAP = CFG["preprocessing"]["chunk_overlap"]
MIN_TEXT_LENGTH = CFG["preprocessing"]["min_text_length"]

# DuckDB chunk table schema
CHUNK_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        VARCHAR PRIMARY KEY,
    doc_id          VARCHAR NOT NULL,
    issuer          VARCHAR,
    borrower        VARCHAR,
    sector          VARCHAR,
    rating_bucket   VARCHAR,
    date            VARCHAR,
    doc_type        VARCHAR,
    event_type      VARCHAR,
    section         VARCHAR NOT NULL,
    chunk_type      VARCHAR NOT NULL,
    chunk_index     INTEGER NOT NULL,
    text            VARCHAR NOT NULL,
    net_leverage    DOUBLE,
    total_cap       DOUBLE,
    esg_score       DOUBLE,
    collateral      VARCHAR,
    recommended_action VARCHAR,
    portfolio_action   VARCHAR,
    watchlist_flag  BOOLEAN,
    spread_change   INTEGER,
    rating_change   INTEGER
)
"""


# ── Text splitter (LangChain chunking module) ─────────────────────────────────
def _build_splitter() -> RecursiveCharacterTextSplitter:
    """Build a LangChain RecursiveCharacterTextSplitter for prose sections.

    Uses a hierarchy of separators that respects paragraph and sentence
    boundaries before falling back to character-level splitting.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


# ── JSONL reader with resume support ─────────────────────────────────────────
def _read_corpus(
    corpus_path: Path,
    limit: int | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Yield documents from the JSONL corpus file.

    Args:
        corpus_path: Path to the JSONL file produced by generate_corpus.py.
        limit: If set, stop after this many documents (for quick test runs).

    Yields:
        Parsed document dicts. Skips and logs malformed lines.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {corpus_path}. "
            "Run generate_corpus.py first."
        )

    count = 0
    with open(corpus_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                yield doc
                count += 1
                if limit and count >= limit:
                    log.info("Reached --limit %d — stopping early.", limit)
                    return
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d: %s", line_num, exc)


# ── Memo-level metadata extractor ────────────────────────────────────────────
def _memo_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    """Extract memo-level metadata fields attached to every chunk.

    These fields allow the retrieval layer to filter any chunk by
    structured attributes without needing a join back to the document.
    """
    return {
        "doc_id":           doc.get("doc_id"),
        "issuer":           doc.get("issuer"),
        "borrower":         doc.get("borrower"),
        "sector":           doc.get("sector"),
        "rating_bucket":    doc.get("rating_bucket"),
        "date":             doc.get("date"),
        "doc_type":         doc.get("doc_type"),
        "event_type":       doc.get("event_type"),
        "net_leverage":     doc.get("net_leverage"),
        "total_cap":        doc.get("total_cap"),
        "esg_score":        doc.get("esg_score"),
        "collateral":       doc.get("collateral"),
        "recommended_action": doc.get("recommended_action"),
        "portfolio_action": doc.get("portfolio_action"),
        "watchlist_flag":   bool(doc.get("watchlist_flag", False)),
        "spread_change":    doc.get("spread_change"),
        "rating_change":    doc.get("rating_change"),
    }


# ── Section mappers ───────────────────────────────────────────────────────────
def _section_executive_header(doc: dict[str, Any]) -> str:
    """Build executive summary header text from structured fields.

    Converts header metadata into embeddable prose so the issuer identity,
    ESG profile, and key metrics are retrievable as a single semantic unit.
    """
    parts = [
        f"{doc.get('issuer', 'Unknown issuer')} is a "
        f"{doc.get('sector', 'unknown sector')} company",
    ]

    if doc.get("borrower"):
        parts[0] += f" (borrower: {doc['borrower']})"

    rating = doc.get("rating_bucket")
    if rating:
        parts[0] += f" rated {rating}."
    else:
        parts[0] += "."

    if doc.get("collateral"):
        collateral_text = doc["collateral"].replace("_", " ")
        parts.append(f"The transaction is secured on a {collateral_text} basis.")

    esg = doc.get("esg_score")
    if esg is not None:
        parts.append(f"ESG score: {esg:.1f}.")

    leverage = doc.get("net_leverage")
    if leverage is not None:
        parts.append(f"Net leverage of {leverage:.1f}x.")

    total_cap = doc.get("total_cap")
    if total_cap is not None:
        parts.append(f"Total capitalization of ${total_cap:,.0f}mm.")

    return " ".join(parts)


def _section_executive_reco(doc: dict[str, Any]) -> str:
    """Build executive summary recommendation text from recommendation fields."""
    action = doc.get("recommended_action", "").replace("_", " ")
    reco = doc.get("recommendations", "")
    watchlist = doc.get("watchlist_flag", False)

    parts = []
    if action:
        parts.append(f"Recommended action: {action}.")
    if reco:
        parts.append(reco)
    if watchlist:
        parts.append("Issuer is on watchlist.")

    return " ".join(parts) if parts else ""


def _section_company_description(doc: dict[str, Any]) -> str:
    """Extract company description text from summary_text field."""
    return doc.get("summary_text", "")


def _section_transaction_overview(doc: dict[str, Any]) -> str:
    """Build transaction overview from headline and full_commentary."""
    headline = doc.get("headline", "")
    commentary = doc.get("full_commentary", "")
    parts = [p for p in [headline, commentary] if p]
    return " ".join(parts)


def _section_merits(doc: dict[str, Any]) -> str:
    """Extract merits portion from full_commentary (first half by sentence count)."""
    commentary = doc.get("full_commentary", "")
    sentences = [s.strip() for s in commentary.split(".") if s.strip()]
    half = max(1, len(sentences) // 2)
    return ". ".join(sentences[:half]) + "." if sentences else ""


def _section_concerns(doc: dict[str, Any]) -> str:
    """Extract concerns portion from full_commentary (second half by sentence count)."""
    commentary = doc.get("full_commentary", "")
    sentences = [s.strip() for s in commentary.split(".") if s.strip()]
    half = max(1, len(sentences) // 2)
    return ". ".join(sentences[half:]) + "." if sentences[half:] else ""


def _section_rep_risk_esg(doc: dict[str, Any]) -> str:
    """Build rep risk and ESG section from ESG score and outcome note."""
    parts = []
    esg = doc.get("esg_score")
    if esg is not None:
        parts.append(f"ESG composite score of {esg:.1f}.")
    outcome = doc.get("outcome_note", "")
    if outcome:
        parts.append(outcome)
    return " ".join(parts)


# ── Section dispatcher ────────────────────────────────────────────────────────
def _extract_sections(
    doc: dict[str, Any],
    serializer: TableSerializer,
) -> list[dict[str, Any]]:
    """Extract all MVP sections from a flat corpus document.

    Returns a list of section dicts, each with keys:
        section:    section identifier string
        chunk_type: 'text' or 'table_serialized'
        text:       the embeddable text content

    Args:
        doc: Flat corpus document from generate_corpus.py.
        serializer: TableSerializer instance for cap table conversion.
    """
    sections = []

    # 1. Executive summary — header
    header_text = _section_executive_header(doc)
    if len(header_text) >= MIN_TEXT_LENGTH:
        sections.append({
            "section":    "executive_summary_header",
            "chunk_type": "text",
            "text":       header_text,
        })

    # 2. Executive summary — recommendation
    reco_text = _section_executive_reco(doc)
    if len(reco_text) >= MIN_TEXT_LENGTH:
        sections.append({
            "section":    "executive_summary_reco",
            "chunk_type": "text",
            "text":       reco_text,
        })

    # 3. Executive summary — cap table (table → serialized text)
    cap_text = serializer.serialize_cap_table(doc)
    if cap_text and len(cap_text) >= MIN_TEXT_LENGTH:
        sections.append({
            "section":    "executive_summary_cap_table",
            "chunk_type": "table_serialized",
            "text":       cap_text,
        })

    # 4. Company description
    desc_text = _section_company_description(doc)
    if len(desc_text) >= MIN_TEXT_LENGTH:
        sections.append({
            "section":    "company_description",
            "chunk_type": "text",
            "text":       desc_text,
        })

    # 5. Transaction overview
    txn_text = _section_transaction_overview(doc)
    if len(txn_text) >= MIN_TEXT_LENGTH:
        sections.append({
            "section":    "transaction_overview",
            "chunk_type": "text",
            "text":       txn_text,
        })

    # 6a. Merits
    merits_text = _section_merits(doc)
    if len(merits_text) >= MIN_TEXT_LENGTH:
        sections.append({
            "section":    "merits_and_concerns",
            "chunk_type": "text",
            "text":       merits_text,
        })

    # 6b. Concerns (opposite semantic polarity — separate chunk)
    concerns_text = _section_concerns(doc)
    if len(concerns_text) >= MIN_TEXT_LENGTH:
        sections.append({
            "section":    "concerns",
            "chunk_type": "text",
            "text":       concerns_text,
        })

    # 7. Rep risk and ESG
    esg_text = _section_rep_risk_esg(doc)
    if len(esg_text) >= MIN_TEXT_LENGTH:
        sections.append({
            "section":    "rep_risk_esg",
            "chunk_type": "text",
            "text":       esg_text,
        })

    return sections


# ── Chunk builder ─────────────────────────────────────────────────────────────
def _build_chunks(
    doc: dict[str, Any],
    sections: list[dict[str, Any]],
    splitter: RecursiveCharacterTextSplitter,
    memo_meta: dict[str, Any],
) -> list[dict[str, Any]]:
    """Split sections into chunks and attach metadata to each chunk.

    Text chunks are split with LangChain's RecursiveCharacterTextSplitter.
    Table-serialized chunks are kept as single chunks (already short).

    Args:
        doc:        Source corpus document.
        sections:   Section dicts from _extract_sections().
        splitter:   LangChain text splitter instance.
        memo_meta:  Memo-level metadata dict from _memo_metadata().

    Returns:
        List of chunk row dicts ready for DuckDB insertion.
    """
    chunks = []
    global_chunk_idx = 0

    for section in sections:
        text = section["text"]
        chunk_type = section["chunk_type"]
        section_name = section["section"]

        if chunk_type == "table_serialized":
            # Keep serialized table as one chunk — already compact prose
            sub_chunks = [text]
        else:
            sub_chunks = splitter.split_text(text)

        for sub_text in sub_chunks:
            if len(sub_text) < MIN_TEXT_LENGTH:
                continue

            chunk_id = f"{doc['doc_id']}_{global_chunk_idx:04d}"

            row = {
                "chunk_id":   chunk_id,
                "section":    section_name,
                "chunk_type": chunk_type,
                "chunk_index": global_chunk_idx,
                "text":       sub_text,
                **memo_meta,
            }
            chunks.append(row)
            global_chunk_idx += 1

    return chunks


# ── DuckDB writer ─────────────────────────────────────────────────────────────
def _init_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open or create the DuckDB database and ensure the chunks table exists.

    Args:
        db_path: Path to the DuckDB file.

    Returns:
        Open DuckDB connection.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute(CHUNK_TABLE_DDL)
    return conn


def _insert_chunks(
    conn: duckdb.DuckDBPyConnection,
    chunks: list[dict[str, Any]],
) -> None:
    """Insert a batch of chunk rows into DuckDB.

    Uses INSERT OR IGNORE semantics so re-runs are idempotent — duplicate
    chunk_ids are silently skipped.

    Args:
        conn:   Open DuckDB connection.
        chunks: List of chunk row dicts from _build_chunks().
    """
    if not chunks:
        return

    df = pd.DataFrame(chunks)

    # Ensure column order matches DDL
    columns = [
        "chunk_id", "doc_id", "issuer", "borrower", "sector",
        "rating_bucket", "date", "doc_type", "event_type",
        "section", "chunk_type", "chunk_index", "text",
        "net_leverage", "total_cap", "esg_score", "collateral",
        "recommended_action", "portfolio_action", "watchlist_flag",
        "spread_change", "rating_change",
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]

    conn.execute(
        "INSERT OR IGNORE INTO chunks SELECT * FROM df"
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────
def normalize(limit: int | None = None) -> None:
    """Run the full preprocessing and chunking pipeline.

    Reads the raw JSONL corpus, maps documents to named sections,
    splits sections into chunks using LangChain, serializes financial
    tables to prose, and writes all chunks to DuckDB.

    Args:
        limit: If set, process only this many documents (for quick runs).
    """
    corpus_path = ROOT / CFG["corpus"]["output_path"]
    db_path = ROOT / CFG["preprocessing"]["processed_db_path"]
    parquet_path = ROOT / CFG["preprocessing"]["processed_parquet_path"]

    log.info("Corpus:  %s", corpus_path)
    log.info("DuckDB:  %s", db_path)
    log.info("Parquet: %s", parquet_path)

    splitter = _build_splitter()
    serializer = TableSerializer(fallback_to_template=True)
    conn = _init_db(db_path)

    total_docs = 0
    total_chunks = 0
    skipped_docs = 0

    try:
        for doc in tqdm(
            _read_corpus(corpus_path, limit=limit),
            desc="Normalizing docs",
            unit="doc",
        ):
            doc_id = doc.get("doc_id")
            if not doc_id:
                skipped_docs += 1
                log.debug("Skipping doc with no doc_id.")
                continue

            memo_meta = _memo_metadata(doc)
            sections = _extract_sections(doc, serializer)

            if not sections:
                skipped_docs += 1
                log.debug("Doc %s produced no sections — skipping.", doc_id)
                continue

            chunks = _build_chunks(doc, sections, splitter, memo_meta)

            if not chunks:
                skipped_docs += 1
                log.debug("Doc %s produced no chunks — skipping.", doc_id)
                continue

            _insert_chunks(conn, chunks)
            total_docs += 1
            total_chunks += len(chunks)

    finally:
        # Export to parquet for downstream use regardless of early exit
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        conn.execute(
            f"COPY chunks TO '{parquet_path}' (FORMAT PARQUET)"
        )
        conn.close()

    log.info(
        "Normalization complete. Docs processed: %d, skipped: %d, "
        "total chunks: %d, avg chunks/doc: %.1f",
        total_docs,
        skipped_docs,
        total_chunks,
        total_chunks / total_docs if total_docs else 0,
    )
    log.info("Chunks written to DuckDB: %s", db_path)
    log.info("Parquet export: %s", parquet_path)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize and chunk the credit memo corpus."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only N documents (useful for quick test runs).",
    )
    args = parser.parse_args()
    normalize(limit=args.limit)
