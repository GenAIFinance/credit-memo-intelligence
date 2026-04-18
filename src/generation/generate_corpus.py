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
    """Build the generation prompt for a given batch size."""
    template = PROMPT_PATH.read_text()
    corpus_cfg = CFG["corpus"]

    return template.format(
        batch_size=batch_size,
        sectors=", ".join(corpus_cfg["sectors"]),
        rating_buckets=", ".join(corpus_cfg["rating_buckets"]),
        doc_types=", ".join(corpus_cfg["doc_types"]),
        event_types=", ".join(corpus_cfg["event_types"]),
    )


def _resolve_deployment() -> str:
    """Resolve Azure deployment name from env or config.

    Config values templated as ${VAR_NAME} are resolved via os.environ.
    Literal config values are used as-is, avoiding brittle string stripping.
    """
    env_override = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if env_override:
        return env_override

    cfg_value = CFG["azure_openai"]["chat_deployment"]
    # Resolve ${VAR_NAME} template syntax
    import re
    match = re.fullmatch(r"\$\{(\w+)\}", cfg_value.strip())
    if match:
        var_name = match.group(1)
        resolved = os.environ.get(var_name)
        if not resolved:
            raise EnvironmentError(
                f"Config references ${{{var_name}}} but that env var is not set."
            )
        return resolved
    # Literal value — use directly
    return cfg_value


# ── Response parsing ──────────────────────────────────────────────────────────
def _parse_response(raw: str, expected_count: int) -> list[dict]:
    """Parse model response into a list of document dicts.

    Accepts either a root JSON array or a dict with a 'documents' key.
    Raises ValueError for any other shape so the caller can retry or skip.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {exc}") from exc

    if isinstance(parsed, list):
        docs = parsed
    elif isinstance(parsed, dict) and "documents" in parsed:
        docs = parsed["documents"]
        if not isinstance(docs, list):
            raise ValueError(
                f"'documents' key present but value is {type(docs)}, expected list."
            )
    else:
        raise ValueError(
            f"Unexpected response shape: expected a JSON array or "
            f"{{\"documents\": [...]}}, got {type(parsed)} with keys "
            f"{list(parsed.keys()) if isinstance(parsed, dict) else 'n/a'}."
        )

    if len(docs) == 0:
        raise ValueError("Model returned an empty document list.")

    log.debug("Parsed %d docs from model (requested %d).", len(docs), expected_count)
    return docs


# ── Single batch call with retry ─────────────────────────────────────────────
# Retry only on transient errors — not on logic/validation bugs
_TRANSIENT_ERRORS = (
    TimeoutError,
    ConnectionError,
)

try:
    from openai import APITimeoutError, APIConnectionError, RateLimitError, InternalServerError
    _TRANSIENT_ERRORS = (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
        TimeoutError,
        ConnectionError,
    )
except ImportError:
    pass  # Fall back to broad base classes if openai version differs


@retry(
    retry=retry_if_exception_type(_TRANSIENT_ERRORS),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
)
def call_azure(client: AzureOpenAI, prompt: str, batch_size: int) -> list[dict]:
    """Call Azure OpenAI and return a parsed list of document dicts.

    Retries only on transient network/rate-limit errors.
    Raises ValueError immediately on bad response shapes (no retry).
    """
    deployment = _resolve_deployment()

    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=CFG["azure_openai"]["max_tokens"] * batch_size,
        temperature=CFG["azure_openai"]["temperature"],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    return _parse_response(raw, expected_count=batch_size)


# ── Safe coercion helpers ─────────────────────────────────────────────────────
_BOOL_TRUE = {True, 1, "1", "true", "yes", "y"}
_BOOL_FALSE = {False, 0, "0", "false", "no", "n", None, ""}


def _to_float(value: object, default: float | None = None) -> float | None:
    """Coerce value to float, returning default on failure instead of raising."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    """Coerce value to int, returning default on failure instead of raising."""
    if value is None:
        return default
    try:
        return int(float(value))  # handles "3.0" → 3
    except (ValueError, TypeError):
        return default


def _to_bool(value: object) -> bool:
    """Parse boolean robustly — treats string 'false'/'no' as False."""
    if isinstance(value, str):
        value = value.strip().lower()
    if value in _BOOL_TRUE:
        return True
    if value in _BOOL_FALSE:
        return False
    # Unknown value — default safe
    return False


# ── Validate and clean a single document ─────────────────────────────────────
REQUIRED_FIELDS = {
    "issuer", "sector", "rating_bucket", "date", "doc_type",
    "event_type", "headline", "summary_text", "full_commentary",
    "recommendations", "recommended_action", "outcome_note", "portfolio_action",
}

# Optional — present but nullable (float fields)
OPTIONAL_NUMERIC = {"net_leverage", "esg_score"}

# Required numeric — always expected, coerce to float
REQUIRED_NUMERIC = {"total_cap"}

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

    # Coerce numerics safely — malformed values get default, not a crash
    doc["exposure_change"] = _to_float(doc.get("exposure_change"), default=0.0)
    doc["spread_change"] = _to_int(doc.get("spread_change"), default=0)
    doc["rating_change"] = _to_int(doc.get("rating_change"), default=0)
    doc["watchlist_flag"] = _to_bool(doc.get("watchlist_flag"))

    # Required numeric fields — drop doc if missing or malformed
    for field in REQUIRED_NUMERIC:
        val = _to_float(doc.get(field))
        if val is None:
            log.debug("Dropping doc — required numeric field missing or malformed: %s", field)
            return None
        doc[field] = val

    # Optional numeric fields — coerce or keep None
    for field in OPTIONAL_NUMERIC:
        doc[field] = _to_float(doc.get(field))

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
MAX_CONSECUTIVE_EMPTY = 5  # abort if this many batches in a row yield 0 valid docs


def generate(target: int | None = None) -> None:
    """Run the corpus generation loop until target valid docs are written.

    Uses a while loop so partial batches (model returns fewer valid docs than
    requested) do not cause the corpus to silently undershoot the target.
    """
    corpus_cfg = CFG["corpus"]
    # Use config default only when caller passed None — not when they passed 0
    if target is None:
        target = corpus_cfg["target_docs"]
    if target <= 0:
        raise ValueError(f"target must be a positive integer, got {target}")

    batch_size = corpus_cfg["batch_size"]
    if batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

    output_path = ROOT / corpus_cfg["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check resume state BEFORE creating the Azure client —
    # a no-op rerun should never fail on missing credentials
    already_done = count_existing(output_path)
    if already_done >= target:
        log.info("Already have %d docs (target %d). Nothing to do.", already_done, target)
        return

    client = build_client()

    log.info(
        "Generating docs (target=%d, already_done=%d, batch_size=%d)",
        target, already_done, batch_size,
    )

    written = already_done
    failed_batches = 0
    consecutive_empty = 0
    batch_idx = 0

    with open(output_path, "a") as out_f:
        with tqdm(total=target - already_done, desc="Generating docs", unit="doc") as pbar:
            while written < target:
                this_batch = min(batch_size, target - written)
                prompt = build_prompt(this_batch)

                try:
                    docs = call_azure(client, prompt, this_batch)
                except ValueError as exc:
                    # Bad response shape — log and skip, do not retry
                    log.error("Batch %d bad response (skipping): %s", batch_idx, exc)
                    failed_batches += 1
                    consecutive_empty += 1
                except Exception as exc:
                    # Transient error exhausted retries
                    log.error("Batch %d failed after retries: %s", batch_idx, exc)
                    failed_batches += 1
                    consecutive_empty += 1
                    time.sleep(10)
                else:
                    batch_written = 0
                    for doc in docs:
                        # Cap writes at remaining quota — model may over-return
                        if written + batch_written >= target:
                            log.debug("Target reached mid-batch — discarding surplus docs.")
                            break
                        clean = validate_doc(doc)
                        if clean:
                            out_f.write(json.dumps(clean) + "\n")
                            batch_written += 1

                    out_f.flush()
                    written += batch_written
                    pbar.update(batch_written)

                    if batch_written == 0:
                        consecutive_empty += 1
                        log.warning(
                            "Batch %d produced 0 valid docs (%d consecutive).",
                            batch_idx, consecutive_empty,
                        )
                    else:
                        consecutive_empty = 0

                    log.debug(
                        "Batch %d: wrote %d docs (total %d/%d)",
                        batch_idx, batch_written, written, target,
                    )

                if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                    log.error(
                        "Aborting: %d consecutive batches produced 0 valid docs. "
                        "Check model output and prompt.",
                        consecutive_empty,
                    )
                    break

                batch_idx += 1

                # Polite pause between batches — avoids rate-limit spikes
                if written < target:
                    time.sleep(1.5)

    log.info(
        "Generation complete. Valid docs written: %d / %d. Failed batches: %d.",
        written, target, failed_batches,
    )
    if written < target:
        log.warning(
            "Corpus is %d docs short of target. Re-run to resume from where it left off.",
            target - written,
        )
    if failed_batches:
        log.warning("%d batches failed.", failed_batches)


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
