"""
src/exploration/llm_explorer.py

Phase 3: LLM-First Exploratory Analysis
-----------------------------------------
The centrepiece of the DB job story. Demonstrates LLM-first exploratory
data analysis: GPT-4o labels a stratified 200-doc sample BEFORE any
embedding or clustering. The labels and taxonomy produced here are used
in Phase 4 to calibrate KMeans clusters against LLM-derived structure.

This is the "prompt prototype" in the pipeline narrative:
  GPT-4o labels 200 docs (cheap, ~$4) → validates taxonomy
  BGE + KMeans scales to 3,000 docs (no per-token cost)

Two-step process:
  Step 1: Per-document labeling — each doc gets a risk_theme, action,
          outcome, and four functional decomposition slots
          (trigger, risk_signal, analyst_stance, forward_view)
  Step 2: Taxonomy synthesis — all 200 label outputs fed to GPT-4o
          to produce a refined, frequency-counted taxonomy

Outputs:
  data/processed/llm_labels.parquet   — 200 labeled docs
  data/processed/taxonomy.json        — refined taxonomy v1

Usage:
    python -m src.exploration.llm_explorer
    python -m src.exploration.llm_explorer --sample 50   # quick test
"""

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from src.utils.llm_client import get_llm_client, get_model_name, get_provider_cfg

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"
PROMPT_PATH = ROOT / "prompts" / "taxonomy_prompt.txt"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

logging.basicConfig(
    level=getattr(logging, CFG["logging"]["level"]),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Transient errors for retry ────────────────────────────────────────────────
_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)
try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
    _TRANSIENT_ERRORS = (
        APIConnectionError, APITimeoutError,
        InternalServerError, RateLimitError,
        TimeoutError, ConnectionError,
    )
except ImportError:
    pass

# ── Safe boolean coercion ─────────────────────────────────────────────────────
_BOOL_TRUE = {True, 1, "1", "true", "yes", "y"}
_BOOL_FALSE = {False, 0, "0", "false", "no", "n", None, ""}


def _to_bool(value: object) -> bool | None:
    """Parse boolean robustly — returns None for unrecognized values.

    LLMs sometimes return JSON booleans as strings (e.g. "false" instead
    of false). Plain bool("false") returns True, which is incorrect.

    Returns:
        True / False for known values.
        None for unrecognized values — caller must decide how to handle.
    """
    if isinstance(value, str):
        value = value.strip().lower()
    if value in _BOOL_TRUE:
        return True
    if value in _BOOL_FALSE:
        return False
    return None  # unknown value — signal malformed output to caller


# ── Prompt template sections ──────────────────────────────────────────────────
_FULL_PROMPT = PROMPT_PATH.read_text()
# Split on the section B marker — section A ends there
_SECTION_A = _FULL_PROMPT.split("## SECTION B")[0].strip()
_SECTION_B = "## SECTION B" + _FULL_PROMPT.split("## SECTION B")[1].strip()


# ── Corpus reader ─────────────────────────────────────────────────────────────
def _load_corpus(corpus_path: Path) -> list[dict[str, Any]]:
    """Load the full corpus from JSONL.

    Args:
        corpus_path: Path to corpus.jsonl from Phase 1.

    Returns:
        List of document dicts. Skips malformed lines.

    Raises:
        FileNotFoundError: If corpus does not exist.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {corpus_path}. Run generate_corpus.py first."
        )

    docs = []
    with open(corpus_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d: %s", line_num, exc)

    log.info("Loaded %d documents from corpus.", len(docs))
    return docs


def _stratified_sample(
    docs: list[dict[str, Any]],
    n: int,
    stratify_by: str = "doc_type",
) -> list[dict[str, Any]]:
    """Draw a stratified sample from the corpus.

    Samples proportionally across strata so all doc_types are represented.
    Falls back to random sample if stratify_by field is missing.

    Args:
        docs:         Full corpus document list.
        n:            Target sample size.
        stratify_by:  Field to stratify on (default: doc_type).

    Returns:
        List of sampled documents, length <= n.
    """
    if n >= len(docs):
        log.info("Sample size %d >= corpus size %d — using full corpus.", n, len(docs))
        return docs

    # Group by stratum
    strata: dict[str, list[dict]] = defaultdict(list)
    unstratified = []
    for doc in docs:
        key = doc.get(stratify_by)
        if key:
            strata[key].append(doc)
        else:
            unstratified.append(doc)

    if not strata:
        log.warning("No '%s' field found — falling back to random sample.", stratify_by)
        return random.sample(docs, min(n, len(docs)))

    # Proportional allocation
    total = len(docs)
    sample: list[dict] = []
    for stratum_docs in strata.values():
        proportion = len(stratum_docs) / total
        stratum_n = max(1, round(n * proportion))
        sample.extend(random.sample(stratum_docs, min(stratum_n, len(stratum_docs))))

    # Trim or top up to exactly n
    random.shuffle(sample)
    if len(sample) > n:
        sample = sample[:n]
    elif len(sample) < n:
        remaining = [d for d in docs if d not in sample]
        sample.extend(random.sample(remaining, min(n - len(sample), len(remaining))))

    log.info(
        "Stratified sample: %d docs across %d %s strata.",
        len(sample), len(strata), stratify_by,
    )
    return sample


# ── Per-document labeling ─────────────────────────────────────────────────────
def _build_label_prompt(doc: dict[str, Any]) -> str:
    """Build the per-document labeling prompt (Section A).

    Args:
        doc: A corpus document dict.

    Returns:
        Formatted prompt string.
    """
    taxonomy = CFG["taxonomy"]
    return _SECTION_A.format(
        risk_themes="\n".join(f"  - {t}" for t in taxonomy["risk_themes"]),
        actions="\n".join(f"  - {a}" for a in taxonomy["actions"]),
        outcomes="\n".join(f"  - {o}" for o in taxonomy["outcomes"]),
        issuer=doc.get("issuer", "Unknown"),
        sector=doc.get("sector", "Unknown"),
        rating_bucket=doc.get("rating_bucket", "Unknown"),
        doc_type=doc.get("doc_type", "Unknown"),
        headline=doc.get("headline", ""),
        summary_text=doc.get("summary_text", ""),
        full_commentary=doc.get("full_commentary", ""),
        recommendations=doc.get("recommendations", ""),
        recommended_action=doc.get("recommended_action", ""),
    )


def _build_synthesis_prompt(
    label_outputs: list[dict[str, Any]],
) -> str:
    """Build the taxonomy synthesis prompt (Section B).

    Args:
        label_outputs: List of per-doc label dicts from _label_doc().

    Returns:
        Formatted synthesis prompt string.
    """
    n_docs = len(label_outputs)
    # Compact JSON — fits more docs in context window
    outputs_str = json.dumps(label_outputs, indent=None, separators=(",", ":"))
    return _SECTION_B.format(
        n_docs=n_docs,
        label_outputs=outputs_str,
    )


@retry(
    retry=retry_if_exception_type(_TRANSIENT_ERRORS),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
)
def _call_llm(client: object, prompt: str, max_tokens: int) -> str:
    """Call the LLM and return raw response text.

    Args:
        client:     LLM client from get_llm_client().
        prompt:     Formatted prompt string.
        max_tokens: Max tokens for this call.

    Returns:
        Raw response content string.
    """
    response = client.chat.completions.create(
        model=get_model_name(),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,  # deterministic labeling
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content.strip()


def _parse_label(raw: str, doc_id: str) -> dict[str, Any] | None:
    """Parse and validate a per-document label response.

    Args:
        raw:    Raw JSON string from the LLM.
        doc_id: Document ID for logging context.

    Returns:
        Validated label dict, or None if parsing fails.
    """
    try:
        label = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning("Label parse failed for doc %s: %s", doc_id, exc)
        return None

    required = {
        "risk_theme", "action", "outcome",
        "trigger", "risk_signal", "analyst_stance", "forward_view",
        "confidence", "ambiguous",
    }
    missing = required - set(label.keys())
    if missing:
        log.warning("Label for doc %s missing keys: %s", doc_id, missing)
        return None

    # Coerce types safely
    try:
        label["confidence"] = float(label["confidence"])
    except (ValueError, TypeError):
        label["confidence"] = 0.0

    ambiguous = _to_bool(label.get("ambiguous", False))
    if ambiguous is None:
        log.warning(
            "Doc %s has unrecognized 'ambiguous' value '%s' — dropping label.",
            doc_id, label.get("ambiguous"),
        )
        return None
    label["ambiguous"] = ambiguous
    label["doc_id"] = doc_id
    return label


def _label_doc(
    client: object,
    doc: dict[str, Any],
    provider_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Label a single document with risk theme and decomposition slots.

    Args:
        client:       LLM client.
        doc:          Corpus document dict.
        provider_cfg: Active provider config for token limits.

    Returns:
        Label dict with risk_theme, action, outcome, and four slots,
        or None if labeling fails.
    """
    doc_id = doc.get("doc_id", "unknown")
    prompt = _build_label_prompt(doc)

    try:
        raw = _call_llm(client, prompt, max_tokens=400)
    except Exception as exc:
        log.error("LLM call failed for doc %s: %s", doc_id, exc)
        return None

    label = _parse_label(raw, doc_id)
    if label:
        # Carry forward key memo metadata for the labels parquet
        label["issuer"] = doc.get("issuer")
        label["sector"] = doc.get("sector")
        label["doc_type"] = doc.get("doc_type")
        label["rating_bucket"] = doc.get("rating_bucket")
        label["net_leverage"] = doc.get("net_leverage")
        label["collateral"] = doc.get("collateral")

    return label


# ── Taxonomy synthesis ────────────────────────────────────────────────────────
def _synthesize_taxonomy(
    client: object,
    labels: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Synthesize a refined taxonomy from all label outputs.

    Args:
        client: LLM client.
        labels: All per-doc label dicts from the labeling step.

    Returns:
        Taxonomy dict with risk_themes, actions, outcomes, analyst_notes,
        or None if synthesis fails.
    """
    log.info("Running taxonomy synthesis on %d labels...", len(labels))
    prompt = _build_synthesis_prompt(labels)

    try:
        raw = _call_llm(client, prompt, max_tokens=2000)
    except Exception as exc:
        log.error("Taxonomy synthesis failed: %s", exc)
        return None

    try:
        taxonomy = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("Taxonomy synthesis parse failed: %s", exc)
        return None

    log.info(
        "Taxonomy v1: %d risk themes, %d actions, %d outcomes. "
        "Ambiguous cases: %d. New labels proposed: %d.",
        len(taxonomy.get("risk_themes", [])),
        len(taxonomy.get("actions", [])),
        len(taxonomy.get("outcomes", [])),
        taxonomy.get("ambiguous_cases", 0),
        len(taxonomy.get("new_labels_proposed", [])),
    )
    return taxonomy


# ── Main pipeline ─────────────────────────────────────────────────────────────
def explore(sample_size: int | None = None) -> None:
    """Run LLM-first exploratory analysis on a stratified corpus sample.

    Step 1: Stratified sample of sample_size docs.
    Step 2: Label each doc with GPT-4o (risk theme + decomposition slots).
    Step 3: Synthesize taxonomy from all labels.
    Step 4: Save llm_labels.parquet and taxonomy.json.

    Args:
        sample_size: Override config sample_size (for quick test runs).
    """
    exp_cfg = CFG["exploration"]
    corpus_cfg = CFG["corpus"]

    n = sample_size or exp_cfg["sample_size"]
    if n <= 0:
        raise ValueError(f"sample_size must be positive, got {n}")

    corpus_path = ROOT / corpus_cfg["output_path"]
    labels_path = ROOT / exp_cfg["labels_output_path"]
    taxonomy_path = ROOT / exp_cfg["taxonomy_output_path"]

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    taxonomy_path.parent.mkdir(parents=True, exist_ok=True)

    # Load corpus and draw stratified sample
    docs = _load_corpus(corpus_path)
    sample = _stratified_sample(docs, n=n, stratify_by="doc_type")

    client = get_llm_client()
    provider_cfg = get_provider_cfg()

    # Step 1 — Per-document labeling
    labels: list[dict[str, Any]] = []
    failed = 0

    log.info("Labeling %d documents with GPT-4o...", len(sample))

    for doc in tqdm(sample, desc="Labeling docs", unit="doc"):
        label = _label_doc(client, doc, provider_cfg)
        if label:
            labels.append(label)
        else:
            failed += 1
        # Polite pause — avoid rate limit spikes
        time.sleep(0.5)

    log.info(
        "Labeling complete. Labeled: %d, failed: %d.",
        len(labels), failed,
    )

    if not labels:
        log.error("No labels produced — cannot synthesize taxonomy. Exiting.")
        return

    # Save labels to parquet
    df = pd.DataFrame(labels)
    df.to_parquet(labels_path, index=False)
    log.info("Labels saved to %s (%d rows).", labels_path, len(df))

    # Step 2 — Taxonomy synthesis
    taxonomy = _synthesize_taxonomy(client, labels)
    if taxonomy:
        with open(taxonomy_path, "w") as f:
            json.dump(taxonomy, f, indent=2)
        log.info("Taxonomy v1 saved to %s.", taxonomy_path)
    else:
        log.error("Taxonomy synthesis failed — labels parquet still saved.")

    # Summary stats
    if labels:
        ambiguous = sum(1 for l in labels if l.get("ambiguous"))
        low_conf = sum(1 for l in labels if l.get("confidence", 1.0) < 0.6)
        log.info(
            "Summary — ambiguous: %d (%.0f%%), low confidence: %d (%.0f%%)",
            ambiguous, 100 * ambiguous / len(labels),
            low_conf, 100 * low_conf / len(labels),
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM-first exploratory analysis on credit memo corpus."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Override sample_size from config (e.g. --sample 20 for quick test).",
    )
    args = parser.parse_args()
    explore(sample_size=args.sample)
