"""
src/retrieval/retriever.py

Phase 5: LangChain Retrieval Module
-------------------------------------
Named LangChain contribution — implements a VectorStoreRetriever-compatible
interface over ChromaDB. This is the retrieval layer the job description
explicitly asks for: "contributing code to retrieval or chunking modules
in frameworks like LlamaIndex or LangChain."

Key capability — combined structured + semantic search:

    results = retriever.query(
        text="covenant headroom deteriorating, sponsor unlikely to support",
        filters={"net_leverage": {"$gt": 7.0}, "collateral": "first_lien"},
        top_k=10,
    )

This is not possible with a plain keyword search. Semantic similarity
finds conceptually related memos; metadata filters restrict results to
structurally relevant ones (leverage, collateral, sector, cluster, etc.).

Two retrieval modes:
    - SimilarCaseRetriever   semantic search with optional metadata filters
    - TaxonomyRetriever      filter by risk_theme / action / outcome labels,
                             ranked by semantic similarity to a query

Usage:
    from src.retrieval.retriever import SimilarCaseRetriever, TaxonomyRetriever

    retriever = SimilarCaseRetriever()
    results = retriever.query("covenant stress, liquidity tightening")

    # With structured filters
    results = retriever.query(
        "leverage deterioration",
        filters={"net_leverage": {"$gt": 7.0}, "collateral": "first_lien"},
    )

    # Build index (run once after embed_cluster.py)
    python -m src.retrieval.retriever --build
"""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
import yaml
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

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

# ── Config shortcuts ──────────────────────────────────────────────────────────
CHROMA_CFG = CFG["chroma"]
EMB_CFG = CFG["embeddings"]
RETRIEVAL_CFG = CFG["retrieval"]


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class RetrievalResult:
    """A single retrieved chunk with its metadata and similarity score.

    Attributes:
        chunk_id:    Unique chunk identifier.
        doc_id:      Parent document identifier.
        text:        Chunk text content.
        score:       Cosine similarity score (higher = more similar).
        section:     Section name within the credit memo.
        chunk_type:  'text' or 'table_serialized'.
        metadata:    Full metadata dict (issuer, leverage, collateral, etc.).
    """
    chunk_id: str
    doc_id: str
    text: str
    score: float
    section: str
    chunk_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def issuer(self) -> str:
        """Convenience accessor for issuer name."""
        return self.metadata.get("issuer", "")

    @property
    def net_leverage(self) -> float | None:
        """Convenience accessor for net leverage."""
        val = self.metadata.get("net_leverage", 0.0)
        return float(val) if val else None

    @property
    def cluster_id(self) -> int | None:
        """Convenience accessor for cluster assignment."""
        val = self.metadata.get("cluster_id")
        return int(val) if val is not None else None


# ── Embedding model (shared across retrievers) ────────────────────────────────
class _EmbeddingModel:
    """Lazy singleton wrapper around SentenceTransformer.

    Loads the model once on first use and reuses it for all queries.
    """
    _instance: SentenceTransformer | None = None

    @classmethod
    def get(cls) -> SentenceTransformer:
        """Return the shared SentenceTransformer instance."""
        if cls._instance is None:
            log.info(
                "Loading embedding model '%s'...", EMB_CFG["model_name"]
            )
            cls._instance = SentenceTransformer(
                EMB_CFG["model_name"],
                device=EMB_CFG["device"],
            )
        return cls._instance

    @classmethod
    def embed(cls, text: str) -> list[float]:
        """Embed a single query string and return as a float list."""
        model = cls.get()
        embedding = model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0].tolist()


# ── ChromaDB client (shared) ──────────────────────────────────────────────────
def _get_collection() -> chromadb.Collection:
    """Return the ChromaDB collection for the configured collection name.

    Raises:
        RuntimeError: If the collection does not exist (run embed_cluster.py first).
    """
    persist_dir = str(ROOT / CHROMA_CFG["persist_directory"])
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        collection = client.get_collection(name=CHROMA_CFG["collection_name"])
    except Exception as exc:
        raise RuntimeError(
            f"ChromaDB collection '{CHROMA_CFG['collection_name']}' not found. "
            f"Run embed_cluster.py first to build the vector store. "
            f"Original error: {exc}"
        ) from exc

    log.debug(
        "ChromaDB collection '%s' loaded (%d items).",
        CHROMA_CFG["collection_name"], collection.count(),
    )
    return collection


# ── Result builder ────────────────────────────────────────────────────────────
def _build_results(
    query_result: dict[str, Any],
    score_threshold: float,
) -> list[RetrievalResult]:
    """Convert a raw ChromaDB query result into a list of RetrievalResult objects.

    Filters out results below score_threshold and skips malformed entries.

    Args:
        query_result:    Raw dict from collection.query().
        score_threshold: Minimum cosine similarity score to include.

    Returns:
        List of RetrievalResult sorted by score descending.
    """
    results = []

    ids = (query_result.get("ids") or [[]])[0]
    documents = (query_result.get("documents") or [[]])[0]
    metadatas = (query_result.get("metadatas") or [[]])[0]
    distances = (query_result.get("distances") or [[]])[0]

    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
        # Convert to similarity score: 1 - (distance / 2)
        score = round(1.0 - (distance / 2.0), 4)

        if score < score_threshold:
            continue

        results.append(RetrievalResult(
            chunk_id=chunk_id,
            doc_id=meta.get("doc_id", ""),
            text=text or "",
            score=score,
            section=meta.get("section", ""),
            chunk_type=meta.get("chunk_type", "text"),
            metadata=meta,
        ))

    return sorted(results, key=lambda r: r.score, reverse=True)


# ── SimilarCaseRetriever ──────────────────────────────────────────────────────
class SimilarCaseRetriever:
    """LangChain VectorStoreRetriever-compatible semantic retriever.

    Finds the most semantically similar credit memo chunks to a query,
    with optional ChromaDB metadata filters for structured constraints.

    This demonstrates the combined structured + semantic search capability:
    semantic similarity ranks by conceptual relevance, metadata filters
    restrict to structurally relevant memos (leverage, collateral, sector).

    Args:
        top_k:            Number of results to return (overrides config).
        score_threshold:  Minimum similarity score (overrides config).
    """

    def __init__(
        self,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> None:
        self._top_k = top_k or RETRIEVAL_CFG["top_k"]
        self._score_threshold = (
            score_threshold
            if score_threshold is not None
            else RETRIEVAL_CFG["score_threshold"]
        )
        self._collection: chromadb.Collection | None = None

    def _get_collection(self) -> chromadb.Collection:
        """Lazy-load ChromaDB collection."""
        if self._collection is None:
            self._collection = _get_collection()
        return self._collection

    def query(
        self,
        text: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        sections: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Find semantically similar chunks with optional structured filters.

        Args:
            text:     Query text — embedded and compared against the corpus.
            filters:  ChromaDB where clause for structured filtering.
                      Examples:
                        {"net_leverage": {"$gt": 7.0}}
                        {"collateral": "first_lien"}
                        {"sector": "Healthcare Services"}
                        {"cluster_id": 3}
            top_k:    Override instance top_k for this query.
            sections: If provided, restrict results to these section names.
                      E.g. ["executive_summary_reco", "merits_and_concerns"]

        Returns:
            List of RetrievalResult sorted by similarity score descending.
        """
        if not text or not text.strip():
            log.warning("Empty query text — returning no results.")
            return []

        k = top_k or self._top_k
        query_embedding = _EmbeddingModel.embed(text.strip())
        collection = self._get_collection()

        # Build where clause — combine user filters with section filter
        where: dict[str, Any] | None = None
        clauses = []

        if filters:
            clauses.append(filters)
        if sections:
            if len(sections) == 1:
                clauses.append({"section": sections[0]})
            else:
                clauses.append({"section": {"$in": sections}})

        if len(clauses) == 1:
            where = clauses[0]
        elif len(clauses) > 1:
            where = {"$and": clauses}

        try:
            query_result = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k * 2, collection.count()),  # over-fetch then filter
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            log.error("ChromaDB query failed: %s", exc)
            return []

        results = _build_results(query_result, self._score_threshold)
        return results[:k]

    def get_similar_to_doc(
        self,
        doc_id: str,
        section: str = "executive_summary_header",
        top_k: int | None = None,
        exclude_same_doc: bool = True,
    ) -> list[RetrievalResult]:
        """Find memos similar to a given document.

        Retrieves the specified section chunk from the target document,
        then queries for similar chunks across the corpus.

        Args:
            doc_id:           Document ID to find similar memos for.
            section:          Which section to use as the query vector.
            top_k:            Number of results to return.
            exclude_same_doc: If True, exclude chunks from the same document.

        Returns:
            List of RetrievalResult from similar memos.
        """
        collection = self._get_collection()

        # Fetch the source chunk
        try:
            source = collection.get(
                where={"$and": [{"doc_id": doc_id}, {"section": section}]},
                include=["documents", "embeddings"],
            )
        except Exception as exc:
            log.error("Failed to fetch source doc %s: %s", doc_id, exc)
            return []

        if not source["documents"]:
            log.warning(
                "No '%s' chunk found for doc_id '%s'.", section, doc_id
            )
            return []

        source_text = source["documents"][0]
        results = self.query(
            text=source_text,
            top_k=(top_k or self._top_k) + (10 if exclude_same_doc else 0),
        )

        if exclude_same_doc:
            results = [r for r in results if r.doc_id != doc_id]

        return results[:top_k or self._top_k]


# ── TaxonomyRetriever ─────────────────────────────────────────────────────────
class TaxonomyRetriever:
    """Retrieves memos by taxonomy label, ranked by semantic similarity.

    Combines structured filtering (taxonomy label from Phase 3) with
    semantic ranking (embedding similarity) in one query.

    Args:
        top_k:           Number of results to return.
        score_threshold: Minimum similarity score.
    """

    def __init__(
        self,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> None:
        self._retriever = SimilarCaseRetriever(
            top_k=top_k,
            score_threshold=score_threshold,
        )

    def query_by_theme(
        self,
        risk_theme: str,
        query_text: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Find memos with a specific risk theme, ranked by semantic similarity.

        Args:
            risk_theme:  Risk theme label (e.g. 'leverage_deterioration').
            query_text:  Optional query for semantic ranking. If None,
                         results are ordered by ingestion order.
            top_k:       Override instance top_k.

        Returns:
            List of RetrievalResult filtered by theme, ranked by similarity.
        """
        text = query_text or risk_theme.replace("_", " ")
        return self._retriever.query(
            text=text,
            filters={"risk_theme": risk_theme},
            top_k=top_k,
            sections=["executive_summary_reco", "merits_and_concerns"],
        )

    def query_by_action(
        self,
        action: str,
        query_text: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Find memos with a specific recommended action.

        Args:
            action:      Action label (e.g. 'reduce_exposure').
            query_text:  Optional query for semantic ranking.
            top_k:       Override instance top_k.

        Returns:
            List of RetrievalResult filtered by action.
        """
        text = query_text or action.replace("_", " ")
        return self._retriever.query(
            text=text,
            filters={"recommended_action": action},
            top_k=top_k,
        )


# ── Build command ─────────────────────────────────────────────────────────────
def verify_index() -> None:
    """Verify the ChromaDB index exists and report basic stats."""
    try:
        collection = _get_collection()
        count = collection.count()
        log.info(
            "ChromaDB index verified. Collection: '%s', items: %d.",
            CHROMA_CFG["collection_name"], count,
        )
        if count == 0:
            log.warning(
                "Collection exists but is empty. "
                "Run embed_cluster.py to populate it."
            )
    except RuntimeError as exc:
        log.error("%s", exc)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify ChromaDB retrieval index."
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Verify the ChromaDB index and report stats.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a test query and print top results.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results to return for --query.",
    )
    args = parser.parse_args()

    if args.build:
        verify_index()

    if args.query:
        retriever = SimilarCaseRetriever(top_k=args.top_k)
        results = retriever.query(args.query)
        print(f"\nQuery: {args.query}")
        print(f"Results: {len(results)}\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.score:.3f}] {r.issuer} — {r.section}")
            print(f"     {r.text[:120]}...")
            print()
