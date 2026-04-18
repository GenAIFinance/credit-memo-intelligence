"""
src/generation/generate_corpus.py

Phase 1: Corpus Generation
--------------------------
Uses Azure OpenAI GPT-4o to generate a synthetic corpus of leveraged loan
credit research documents. Writes incrementally to JSONL so progress is
never lost on a crash or rate-limit pause.

Usage:
    python -m src.generation.generate_corpus
    python -m src.generation.generate_corpus --target 500   # quick test run
"""

import argparse
import json
import logging
import os
import time
import uuid
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import AzureOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"
PROMPT_PATH = ROOT / "prompts" / "generation_prompt.txt"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

logging.basicConfig(
    level=getattr(logging, CFG["logging"]["level"]),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ── Azure OpenAI client ───────────────────────────────────────────────────────
def build_client() -> AzureOpenAI:
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    api_version = CFG["azure_openai"]["api_version"]

    if not endpoint or not api_key:
        raise EnvironmentError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set. "
            "Copy .env.example to .env and fill in your values."
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_prompt(batch_size: int) -> str:
    template = PROMPT_PATH.read_text()
    az_cfg = CFG["azure_openai"]
    corpus_cfg = CFG["corpus"]

    return template.format(
        batch_size=batch_size,
        sectors=", ".join(corpus_cfg["sectors"]),
        rating_buckets=", ".join(corpus_cfg["rating_buckets"]),
        doc_types=", ".join(corpus_cfg["doc_types"]),
        event_types=", ".join(corpus_cfg["event_types"]),
    )


# ── Single batch call with retry ─────────────────────────────────────────────
@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
)
def call_azure(client: AzureOpenAI, prompt: str, batch_size: int) -> list[dict]:
    """Call Azure OpenAI and parse the returned JSON array."""
    deployment = os.environ.get(
        "AZURE_OPENAI_DEPLOYMENT",
        CFG["azure_openai"]["chat_deployment"].strip("${}"),
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=CFG["azure_openai"]["max_tokens"] * batch_size,
        temperature=CFG["azure_openai"]["temperature"],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()

    # Model sometimes wraps in {"documents": [...]} — unwrap defensively
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        # Find the first list value
        for v in parsed.values():
            if isinstance(v, list):
                parsed = v
                break

    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON array, got: {type(parsed)}")

    return parsed


# ── Validate and clean a single document ─────────────────────────────────────
REQUIRED_FIELDS = {
    "issuer", "sector", "rating_bucket", "date", "doc_type",
    "event_type", "headline", "summary_text", "full_commentary",
    "recommendations", "recommended_action", "outcome_note", "portfolio_action",
}

# Optional — present but nullable (float fields)
OPTIONAL_NUMERIC = {"net_leverage", "market_cap", "esg_score"}

VALID_ACTIONS = {
    "maintain_position", "reduce_exposure", "add_on_weakness",
    "escalate_for_review", "place_on_watchlist", "seek_more_diligence",
    "reassess_thesis", "exit",
}

VALID_PORTFOLIO = {"hold", "trim", "add", "sell", "monitor"}

VALID_COLLATERAL = {
    "first_lien", "second_lien", "unsecured",
    "first_lien_second_lien_split", "super_senior_revolver",
}


def validate_doc(doc: dict) -> dict | None:
    """Return cleaned doc or None if fatally malformed."""
    if not isinstance(doc, dict):
        return None

    # Check required fields present and non-empty
    for field in REQUIRED_FIELDS:
        if not doc.get(field):
            log.debug("Dropping doc missing field: %s", field)
            return None

    # Enforce controlled vocabularies
    if doc.get("recommended_action") not in VALID_ACTIONS:
        doc["recommended_action"] = "maintain_position"

    if doc.get("portfolio_action") not in VALID_PORTFOLIO:
        doc["portfolio_action"] = "monitor"

    if doc.get("collateral") not in VALID_COLLATERAL:
        doc["collateral"] = None

    # Coerce numerics safely
    doc["exposure_change"] = float(doc.get("exposure_change") or 0.0)
    doc["spread_change"] = int(doc.get("spread_change") or 0)
    doc["rating_change"] = int(doc.get("rating_change") or 0)
    doc["watchlist_flag"] = bool(doc.get("watchlist_flag", False))

    # Optional numeric fields — coerce to float or keep None
    for field in OPTIONAL_NUMERIC:
        val = doc.get(field)
        doc[field] = float(val) if val is not None else None

    # Guarantee a doc_id
    if not doc.get("doc_id"):
        doc["doc_id"] = str(uuid.uuid4())

    return doc


# ── Resume support — count already-written docs ───────────────────────────────
def count_existing(output_path: Path) -> int:
    if not output_path.exists():
        return 0
    with open(output_path) as f:
        return sum(1 for line in f if line.strip())


# ── Main generation loop ──────────────────────────────────────────────────────
def generate(target: int | None = None) -> None:
    corpus_cfg = CFG["corpus"]
    target = target or corpus_cfg["target_docs"]
    batch_size = corpus_cfg["batch_size"]
    output_path = ROOT / corpus_cfg["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = build_client()
    already_done = count_existing(output_path)

    if already_done >= target:
        log.info("Already have %d docs (target %d). Nothing to do.", already_done, target)
        return

    remaining = target - already_done
    log.info(
        "Generating %d docs (target=%d, already_done=%d, batch_size=%d)",
        remaining, target, already_done, batch_size,
    )

    n_batches = (remaining + batch_size - 1) // batch_size
    written = already_done
    failed_batches = 0

    with open(output_path, "a") as out_f:
        with tqdm(total=remaining, desc="Generating docs", unit="doc") as pbar:
            for batch_idx in range(n_batches):
                # Last batch may be smaller
                this_batch = min(batch_size, target - written)
                prompt = build_prompt(this_batch)

                try:
                    docs = call_azure(client, prompt, this_batch)
                except Exception as exc:
                    log.error("Batch %d failed after retries: %s", batch_idx, exc)
                    failed_batches += 1
                    time.sleep(10)
                    continue

                batch_written = 0
                for doc in docs:
                    clean = validate_doc(doc)
                    if clean:
                        out_f.write(json.dumps(clean) + "\n")
                        batch_written += 1

                out_f.flush()
                written += batch_written
                pbar.update(batch_written)

                log.debug(
                    "Batch %d/%d: wrote %d docs (total %d/%d)",
                    batch_idx + 1, n_batches, batch_written, written, target,
                )

                # Polite pause between batches — avoids rate-limit spikes
                if batch_idx < n_batches - 1:
                    time.sleep(1.5)

    log.info(
        "Generation complete. Total docs written: %d. Failed batches: %d.",
        written, failed_batches,
    )
    if failed_batches:
        log.warning(
            "%d batches failed. Re-run the script to resume — it will pick up where it left off.",
            failed_batches,
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic credit corpus")
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Override target doc count from config (useful for quick test runs)",
    )
    args = parser.parse_args()
    generate(target=args.target)
