"""
src/preprocessing/table_serializer.py

LangChain-compatible module that converts structured financial table data
into embeddable analyst prose using Azure OpenAI GPT-4o.

This is a named LangChain contribution module — it implements the
'chunking module' pattern referenced in the job description by treating
table serialization as a pre-chunking transformation step.

Usage:
    from src.preprocessing.table_serializer import TableSerializer
    serializer = TableSerializer()
    prose = serializer.serialize(table_data, table_context)
"""

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"
PROMPT_PATH = ROOT / "prompts" / "table_serialization_prompt.txt"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

log = logging.getLogger(__name__)

# ── LLM client — provided by src.utils.llm_client ───────────────────────────
from src.utils.llm_client import get_llm_client, get_model_name, get_provider_cfg

# Transient error types for retry
_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)
try:
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    _TRANSIENT_ERRORS = (
        APIConnectionError, APITimeoutError, InternalServerError,
        RateLimitError, TimeoutError, ConnectionError,
    )
except ImportError:
    pass


def _format_table_data(table_data: dict[str, Any]) -> str:
    """Format a dict of table key-value pairs into a readable string for the prompt."""
    lines = []
    for key, value in table_data.items():
        if value is not None:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


class TableSerializer:
    """Converts financial table data to embeddable analyst prose.

    Acts as a LangChain-compatible pre-chunking transformation module.
    Serializes structured numeric/categorical table data into fluent
    analyst prose that can be embedded alongside narrative text chunks.

    Args:
        max_retries: Number of retry attempts on transient API errors.
        fallback_to_template: If True, return a template-based string
            when the API call fails, rather than raising. Useful for
            batch processing where a single table failure should not
            abort the pipeline.
    """

    def __init__(
        self,
        max_retries: int = 3,
        fallback_to_template: bool = True,
    ) -> None:
        self._max_retries = max_retries
        self._fallback_to_template = fallback_to_template
        self._prompt_template = PROMPT_PATH.read_text()
        self._client: object | None = None

    def _get_client(self) -> object:
        """Lazy-initialize the LLM client via provider abstraction layer."""
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    @retry(
        retry=retry_if_exception_type(_TRANSIENT_ERRORS),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    def _call_api(self, prompt: str) -> str:
        """Call the configured LLM with retry on transient errors only."""
        client = self._get_client()

        response = client.chat.completions.create(
            model=get_model_name(),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,  # low temperature — factual serialization
        )
        return response.choices[0].message.content.strip()

    def _build_template_fallback(
        self,
        table_data: dict[str, Any],
        table_context: str,
    ) -> str:
        """Build a minimal template-based prose string as API fallback.

        Ensures the pipeline can continue even if the serialization API
        call fails — produces a less fluent but machine-readable string.
        """
        parts = [f"Financial data for {table_context}."]
        for key, value in table_data.items():
            if value is not None:
                parts.append(f"{key} is {value}.")
        return " ".join(parts)

    def serialize(
        self,
        table_data: dict[str, Any],
        table_context: str = "credit memo",
    ) -> str:
        """Serialize a financial table dict to embeddable analyst prose.

        Args:
            table_data: Dict of field names to values extracted from the
                financial table (e.g. {"Net Leverage": "6.8x",
                "Total Cap": "$3,200mm", "LTM EBITDA": "$471mm"}).
            table_context: Human-readable description of where this table
                appears (e.g. "capitalization table", "summary of terms").

        Returns:
            Analyst prose string suitable for embedding. Falls back to a
            template string if the API call fails and fallback_to_template
            is True.

        Raises:
            Exception: Re-raises API exceptions when fallback_to_template
                is False.
        """
        if not table_data:
            log.warning("serialize() called with empty table_data — returning empty string.")
            return ""

        formatted = _format_table_data(table_data)
        prompt = self._prompt_template.format(
            table_context=table_context,
            table_data=formatted,
        )

        try:
            prose = self._call_api(prompt)
            log.debug("Serialized table (%s) → %d chars", table_context, len(prose))
            return prose
        except Exception as exc:
            log.error(
                "Table serialization failed for context '%s': %s",
                table_context, exc,
            )
            if self._fallback_to_template:
                fallback = self._build_template_fallback(table_data, table_context)
                log.warning(
                    "Using template fallback for '%s' (%d chars).",
                    table_context, len(fallback),
                )
                return fallback
            raise

    def serialize_cap_table(self, doc: dict[str, Any]) -> str:
        """Serialize the capitalization table fields from a corpus document.

        Extracts net_leverage, total_cap, spread_change, and rating_bucket
        from a flat corpus document dict and serializes them as a cap table.

        Args:
            doc: A flat corpus document dict as produced by generate_corpus.py.

        Returns:
            Analyst prose describing the capitalization structure.
        """
        table_data: dict[str, Any] = {}

        if doc.get("net_leverage") is not None:
            table_data["Net Leverage"] = f"{doc['net_leverage']:.1f}x"

        if doc.get("total_cap") is not None:
            table_data["Total Capitalization"] = f"${doc['total_cap']:,.0f}mm"

        if doc.get("collateral"):
            table_data["Security"] = doc["collateral"].replace("_", " ").title()

        if doc.get("rating_bucket"):
            table_data["Rating"] = doc["rating_bucket"]

        if doc.get("spread_change") is not None and doc["spread_change"] != 0:
            direction = "widened" if doc["spread_change"] > 0 else "tightened"
            table_data["Spread Movement"] = f"{abs(doc['spread_change'])}bps {direction}"

        if doc.get("esg_score") is not None:
            table_data["ESG Score"] = f"{doc['esg_score']:.1f}"

        return self.serialize(
            table_data=table_data,
            table_context="capitalization table",
        )
