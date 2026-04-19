"""
src/embeddings/embed_cluster.py

Phase 4: Embeddings + Semantic Clustering
------------------------------------------
This is the "scaling prompt prototypes into embedding pipelines" step —
the core technical narrative for the DB role.

GPT-4o labeled 200 docs in Phase 3 (cheap prototype). Here we:
  1. Embed all chunks from DuckDB using BAAI/bge-small-en-v1.5 (local, free)
  2. Cluster embeddings with KMeans
  3. Calibrate cluster labels against GPT-4o labels (cluster purity)
  4. Reduce to 2D with UMAP for the cluster explorer UI
  5. Write embeddings, cluster IDs, and UMAP coords back to DuckDB
  6. Persist vector store in ChromaDB for Phase 5 retrieval

Cluster purity is the interview-ready metric:
  "GPT-4o labeled 200 docs at ~$4 total. We used those labels to
   validate that BGE + KMeans captures the same semantic structure —
   cluster purity of X% means the embedding space agrees with the
   LLM's taxonomy. We then scaled to 3,000 docs at zero marginal cost."

Usage:
    python -m src.embeddings.embed_cluster
    python -m src.embeddings.embed_cluster --dry-run   # skip ChromaDB write
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import chromadb
import duckdb
import numpy as np
import pandas as pd
import yaml
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from umap import UMAP

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
EMB_CFG = CFG["embeddings"]
CHROMA_CFG = CFG["chroma"]


# ── Chunk loader ──────────────────────────────────────────────────────────────
def _load_chunks(db_path: Path) -> pd.DataFrame:
    """Load all chunks from DuckDB into a DataFrame.

    Args:
        db_path: Path to the DuckDB file from Phase 2.

    Returns:
        DataFrame with chunk_id, text, section, and all memo metadata.

    Raises:
        FileNotFoundError: If DuckDB file does not exist.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"DuckDB not found at {db_path}. Run normalize.py first."
        )

    conn = duckdb.connect(str(db_path), read_only=True)
    df = conn.execute("SELECT * FROM chunks ORDER BY doc_id, chunk_index").df()
    conn.close()

    log.info("Loaded %d chunks from DuckDB.", len(df))
    return df


# ── Embedding ─────────────────────────────────────────────────────────────────
def _embed_chunks(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Embed a list of text chunks using a SentenceTransformer model.

    Args:
        texts:      List of text strings to embed.
        model_name: HuggingFace model identifier.
        batch_size: Encoding batch size.
        device:     'cpu' or 'cuda'.

    Returns:
        2D numpy array of shape (n_chunks, embedding_dim).
    """
    log.info(
        "Loading embedding model '%s' on %s...", model_name, device
    )
    model = SentenceTransformer(model_name, device=device)

    log.info("Embedding %d chunks (batch_size=%d)...", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity via dot product
        device=device,
    )
    log.info(
        "Embeddings complete. Shape: %s, dtype: %s",
        embeddings.shape, embeddings.dtype,
    )
    return embeddings


# ── KMeans clustering ─────────────────────────────────────────────────────────
def _cluster(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """Fit KMeans on embeddings and return cluster label array.

    Args:
        embeddings:   2D embedding array (n_chunks, dim).
        n_clusters:   Number of KMeans clusters from config.
        random_state: Seed for reproducibility.

    Returns:
        1D integer array of cluster assignments (n_chunks,).
    """
    log.info("Fitting KMeans (n_clusters=%d)...", n_clusters)
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )
    labels = km.fit_predict(embeddings)
    log.info("KMeans complete. Inertia: %.2f", km.inertia_)
    return labels


# ── UMAP reduction ────────────────────────────────────────────────────────────
def _reduce_umap(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    n_components: int,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embeddings to 2D with UMAP for cluster explorer UI.

    Args:
        embeddings:   2D embedding array.
        n_neighbors:  UMAP n_neighbors param — controls local vs global structure.
        min_dist:     UMAP min_dist param — controls cluster compactness.
        n_components: Target dimensionality (2 for UI scatter plot).
        random_state: Seed for reproducibility.

    Returns:
        2D array of shape (n_chunks, n_components).
    """
    log.info(
        "Running UMAP (n_neighbors=%d, min_dist=%.2f, n_components=%d)...",
        n_neighbors, min_dist, n_components,
    )
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        metric="cosine",
    )
    coords = reducer.fit_transform(embeddings)
    log.info("UMAP complete. Output shape: %s", coords.shape)
    return coords


# ── Cluster purity calibration ────────────────────────────────────────────────
def _compute_cluster_purity(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    llm_labels_path: Path,
) -> dict[str, Any]:
    """Calibrate KMeans clusters against GPT-4o labels from Phase 3.

    Computes per-cluster purity and adjusted Rand index — the key
    metric showing that embedding-based clustering captures the same
    semantic structure as LLM labeling.

    Args:
        df:               Chunk DataFrame with doc_id column.
        cluster_labels:   KMeans cluster assignments.
        llm_labels_path:  Path to llm_labels.parquet from Phase 3.

    Returns:
        Dict with purity scores, ARI, and per-cluster breakdown.
        Returns empty dict if llm_labels.parquet does not exist.
    """
    if not llm_labels_path.exists():
        log.warning(
            "llm_labels.parquet not found at %s — skipping purity calibration. "
            "Run llm_explorer.py first to generate GPT-4o labels.",
            llm_labels_path,
        )
        return {}

    llm_df = pd.read_parquet(llm_labels_path)
    log.info(
        "Loaded %d GPT-4o labels for purity calibration.", len(llm_df)
    )

    # Join cluster assignments to LLM labels on doc_id
    # Use one chunk per doc (executive_summary_header) for a clean join
    chunk_clusters = pd.DataFrame({
        "doc_id": df["doc_id"],
        "cluster": cluster_labels,
        "section": df["section"],
    })
    header_chunks = chunk_clusters[
        chunk_clusters["section"] == "executive_summary_header"
    ].drop_duplicates("doc_id")

    merged = llm_df.merge(header_chunks, on="doc_id", how="inner")
    if merged.empty:
        log.warning("No overlap between LLM labels and chunk doc_ids — skipping purity.")
        return {}

    # Encode risk_theme labels to integers for ARI
    le = LabelEncoder()
    true_labels = le.fit_transform(merged["risk_theme"].fillna("unknown"))
    pred_labels = merged["cluster"].values

    ari = adjusted_rand_score(true_labels, pred_labels)

    # Per-cluster purity: fraction of most common true label in each cluster
    purity_scores = []
    for cluster_id in sorted(merged["cluster"].unique()):
        cluster_mask = merged["cluster"] == cluster_id
        cluster_true = merged.loc[cluster_mask, "risk_theme"]
        if len(cluster_true) == 0:
            continue
        dominant = cluster_true.value_counts().iloc[0]
        purity = dominant / len(cluster_true)
        purity_scores.append({
            "cluster_id": int(cluster_id),
            "size": int(len(cluster_true)),
            "dominant_theme": cluster_true.value_counts().index[0],
            "purity": round(float(purity), 3),
        })

    mean_purity = float(np.mean([p["purity"] for p in purity_scores])) if purity_scores else 0.0

    result = {
        "adjusted_rand_index": round(float(ari), 4),
        "mean_cluster_purity": round(mean_purity, 4),
        "n_labeled_docs": int(len(merged)),
        "per_cluster": purity_scores,
    }

    log.info(
        "Cluster purity: mean=%.3f, ARI=%.4f (across %d labeled docs)",
        mean_purity, ari, len(merged),
    )
    return result


# ── DuckDB writer ─────────────────────────────────────────────────────────────
def _write_clusters_to_db(
    db_path: Path,
    chunk_ids: list[str],
    cluster_labels: np.ndarray,
    umap_coords: np.ndarray,
) -> None:
    """Write cluster IDs and UMAP coordinates back to DuckDB chunks table.

    Adds columns cluster_id, umap_x, umap_y to the chunks table.
    Uses ALTER TABLE ADD COLUMN IF NOT EXISTS for idempotent reruns.

    Args:
        db_path:        Path to DuckDB file.
        chunk_ids:      Ordered list of chunk_id strings matching arrays.
        cluster_labels: KMeans cluster assignments.
        umap_coords:    UMAP 2D coordinates array.
    """
    conn = duckdb.connect(str(db_path))

    # Add columns if not present
    for col, dtype in [
        ("cluster_id", "INTEGER"),
        ("umap_x", "DOUBLE"),
        ("umap_y", "DOUBLE"),
    ]:
        try:
            conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} {dtype}")
        except Exception:
            pass  # Column already exists — idempotent

    # Build update DataFrame
    update_df = pd.DataFrame({
        "chunk_id": chunk_ids,
        "cluster_id": cluster_labels.astype(int),
        "umap_x": umap_coords[:, 0].astype(float),
        "umap_y": umap_coords[:, 1].astype(float),
    })

    conn.execute("""
        UPDATE chunks
        SET cluster_id = u.cluster_id,
            umap_x     = u.umap_x,
            umap_y     = u.umap_y
        FROM update_df u
        WHERE chunks.chunk_id = u.chunk_id
    """)
    conn.close()
    log.info("Cluster IDs and UMAP coords written to DuckDB.")


# ── ChromaDB writer ───────────────────────────────────────────────────────────
def _write_to_chroma(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    risk_theme_map: dict[str, str] | None = None,
) -> None:
    """Persist chunks and embeddings in ChromaDB for Phase 5 retrieval.

    Each chunk is stored with its embedding and a metadata dict containing
    all memo-level metrics plus cluster_id and risk_theme. This enables
    combined semantic + structured filtering at retrieval time.

    Args:
        df:              Chunk DataFrame from DuckDB.
        embeddings:      2D embedding array aligned with df rows.
        cluster_labels:  KMeans cluster assignments.
        risk_theme_map:  Optional dict mapping doc_id to GPT-4o risk_theme
                         label from Phase 3 llm_labels.parquet. Chunks from
                         docs not in the map get risk_theme="".
    """
    persist_dir = str(ROOT / CHROMA_CFG["persist_directory"])
    collection_name = CHROMA_CFG["collection_name"]

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    log.info(
        "Writing %d chunks to ChromaDB collection '%s'...",
        len(df), collection_name,
    )

    # Build metadata — ChromaDB requires scalar values only
    _NULLABLE_STR_COLS = [
        "issuer", "borrower", "sector", "rating_bucket", "date",
        "doc_type", "event_type", "section", "chunk_type",
        "collateral", "recommended_action", "portfolio_action",
    ]
    _NULLABLE_NUM_COLS = [
        "net_leverage", "total_cap", "esg_score",
        "spread_change", "rating_change", "chunk_index",
    ]

    batch_size = 500
    total = len(df)

    for start in tqdm(range(0, total, batch_size), desc="Writing to ChromaDB"):
        end = min(start + batch_size, total)
        batch_df = df.iloc[start:end]
        batch_emb = embeddings[start:end]
        batch_clusters = cluster_labels[start:end]

        ids = batch_df["chunk_id"].tolist()
        documents = batch_df["text"].tolist()
        emb_list = batch_emb.tolist()

        metadatas = []
        for i, row in enumerate(batch_df.itertuples(index=False)):
            meta: dict[str, Any] = {
                "doc_id":     str(row.doc_id),
                "cluster_id": int(batch_clusters[i]),
                "watchlist_flag": bool(getattr(row, "watchlist_flag", False)),
            }
            # risk_theme from Phase 3 GPT-4o labels — join by doc_id
            meta["risk_theme"] = (
                risk_theme_map.get(str(row.doc_id), "")
                if risk_theme_map else ""
            )
            for col in _NULLABLE_STR_COLS:
                val = getattr(row, col, None)
                meta[col] = str(val) if val is not None else ""
            for col in _NULLABLE_NUM_COLS:
                val = getattr(row, col, None)
                # pandas represents NULL numeric DuckDB fields as NaN —
                # val is not None passes for NaN, but float(NaN) produces
                # a non-finite value that ChromaDB rejects or drops silently.
                try:
                    f = float(val)
                    meta[col] = f if math.isfinite(f) else 0.0
                except (TypeError, ValueError):
                    meta[col] = 0.0
            metadatas.append(meta)

        collection.upsert(
            ids=ids,
            embeddings=emb_list,
            documents=documents,
            metadatas=metadatas,
        )

    log.info(
        "ChromaDB write complete. Collection '%s' now has %d items.",
        collection_name, collection.count(),
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────
def embed_and_cluster(dry_run: bool = False) -> None:
    """Run the full embedding and clustering pipeline.

    Steps:
        1. Load chunks from DuckDB (Phase 2 output)
        2. Embed with BGE-small-en-v1.5
        3. Cluster with KMeans
        4. Reduce to 2D with UMAP
        5. Calibrate against GPT-4o labels (cluster purity)
        6. Write cluster IDs + UMAP coords back to DuckDB
        7. Persist in ChromaDB (unless dry_run)
        8. Save embeddings array to disk

    Args:
        dry_run: If True, skip ChromaDB write (useful for testing).
    """
    db_path = ROOT / CFG["preprocessing"]["processed_db_path"]
    emb_path = ROOT / EMB_CFG["embeddings_path"]
    cluster_path = ROOT / EMB_CFG["cluster_labels_path"]
    llm_labels_path = ROOT / CFG["exploration"]["labels_output_path"]

    emb_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1 — Load chunks
    df = _load_chunks(db_path)
    if df.empty:
        log.error("No chunks found in DuckDB. Run normalize.py first.")
        return

    texts = df["text"].tolist()
    chunk_ids = df["chunk_id"].tolist()

    # Step 2 — Embed
    embeddings = _embed_chunks(
        texts=texts,
        model_name=EMB_CFG["model_name"],
        batch_size=EMB_CFG["batch_size"],
        device=EMB_CFG["device"],
    )

    # Save embeddings to disk
    np.save(str(emb_path), embeddings)
    log.info("Embeddings saved to %s.", emb_path)

    # Step 3 — Cluster
    cluster_labels = _cluster(
        embeddings=embeddings,
        n_clusters=EMB_CFG["n_clusters"],
    )

    # Step 4 — UMAP
    umap_coords = _reduce_umap(
        embeddings=embeddings,
        n_neighbors=EMB_CFG["umap_n_neighbors"],
        min_dist=EMB_CFG["umap_min_dist"],
        n_components=EMB_CFG["umap_n_components"],
    )

    # Step 5 — Cluster purity calibration
    purity = _compute_cluster_purity(df, cluster_labels, llm_labels_path)
    if purity:
        purity_path = ROOT / "data/processed/cluster_purity.json"
        with open(purity_path, "w") as f:
            json.dump(purity, f, indent=2)
        log.info("Cluster purity saved to %s.", purity_path)

    # Step 6 — Write back to DuckDB
    _write_clusters_to_db(db_path, chunk_ids, cluster_labels, umap_coords)

    # Save cluster labels parquet
    cluster_df = pd.DataFrame({
        "chunk_id":   chunk_ids,
        "cluster_id": cluster_labels.astype(int),
        "umap_x":     umap_coords[:, 0],
        "umap_y":     umap_coords[:, 1],
    })
    cluster_df.to_parquet(str(cluster_path), index=False)
    log.info("Cluster labels saved to %s.", cluster_path)

    # Step 7 — ChromaDB
    # Build risk_theme map from Phase 3 labels to persist at index time
    risk_theme_map: dict[str, str] = {}
    if llm_labels_path.exists():
        try:
            labels_df = pd.read_parquet(llm_labels_path)
            risk_theme_map = (
                labels_df.dropna(subset=["doc_id", "risk_theme"])
                .set_index("doc_id")["risk_theme"]
                .to_dict()
            )
            log.info(
                "Loaded %d risk_theme labels for ChromaDB indexing.",
                len(risk_theme_map),
            )
        except Exception as exc:
            log.warning("Could not load llm_labels for risk_theme map: %s", exc)

    if dry_run:
        log.info("dry-run mode — skipping ChromaDB write.")
    else:
        _write_to_chroma(df, embeddings, cluster_labels, risk_theme_map=risk_theme_map)

    log.info("Phase 4 complete.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed chunks and cluster with KMeans + UMAP."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip ChromaDB write — useful for testing embedding + clustering only.",
    )
    args = parser.parse_args()
    embed_and_cluster(dry_run=args.dry_run)
