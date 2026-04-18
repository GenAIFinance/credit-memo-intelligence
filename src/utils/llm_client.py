"""
src/utils/llm_client.py

LLM Provider Abstraction Layer
-------------------------------
Single source of truth for LLM client construction. All modules import
from here — no Azure or OpenAI client logic lives anywhere else.

To switch providers, change config/config.yaml:
    llm:
      provider: "openai"   # or "azure"

No code changes required in any other file.

Public API:
    get_llm_client()    → openai.OpenAI or openai.AzureOpenAI
    get_model_name()    → str (model name or Azure deployment name)
    get_provider_cfg()  → dict (active provider config for max_tokens, temperature etc)
"""

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

log = logging.getLogger(__name__)

# ── Supported providers ───────────────────────────────────────────────────────
_SUPPORTED_PROVIDERS = {"openai", "azure"}


def _get_provider() -> str:
    """Return the active provider name from config, validated.

    Raises:
        ValueError: If the provider is not one of the supported options.
    """
    provider = CFG.get("llm", {}).get("provider", "openai").strip().lower()
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. "
            f"Must be one of: {_SUPPORTED_PROVIDERS}. "
            f"Check llm.provider in config/config.yaml."
        )
    return provider


def _resolve_env_template(value: str) -> str:
    """Resolve a ${VAR_NAME} template string via os.environ.

    Passes through literal values unchanged.

    Args:
        value: Config string, either a literal or a ${VAR_NAME} template.

    Returns:
        Resolved string value.

    Raises:
        EnvironmentError: If the referenced env var is not set.
    """
    match = re.fullmatch(r"\$\{(\w+)\}", value.strip())
    if match:
        var_name = match.group(1)
        resolved = os.environ.get(var_name)
        if not resolved:
            raise EnvironmentError(
                f"Config references ${{{var_name}}} but that env var is not set. "
                f"Add it to your .env file."
            )
        return resolved
    return value


# ── Provider-specific builders ────────────────────────────────────────────────
def _build_openai_client() -> OpenAI:
    """Build and return an authenticated OpenAI client.

    Reads OPENAI_API_KEY from env (preferred) or config.

    Raises:
        EnvironmentError: If the API key cannot be resolved.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Fall back to config value (may itself be a ${VAR} template)
        cfg_key = CFG.get("openai", {}).get("api_key", "")
        api_key = _resolve_env_template(cfg_key) if cfg_key else ""

    if not api_key:
        raise EnvironmentError(
            "OpenAI API key not found. Set OPENAI_API_KEY in your .env file."
        )

    log.debug("Building OpenAI client (model: %s)", CFG["openai"]["model"])
    return OpenAI(api_key=api_key)


def _build_azure_client() -> AzureOpenAI:
    """Build and return an authenticated AzureOpenAI client.

    Reads credentials from env vars (preferred) or config templates.

    Raises:
        EnvironmentError: If endpoint or API key cannot be resolved.
    """
    az_cfg = CFG.get("azure_openai", {})

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or \
        _resolve_env_template(az_cfg.get("endpoint", ""))
    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or \
        _resolve_env_template(az_cfg.get("api_key", ""))
    api_version = az_cfg.get("api_version", "2024-02-01")

    if not endpoint or not api_key:
        raise EnvironmentError(
            "Azure OpenAI credentials not found. "
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your .env file."
        )

    log.debug("Building AzureOpenAI client (endpoint: %s)", endpoint)
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


# ── Public API ────────────────────────────────────────────────────────────────
def get_llm_client() -> OpenAI | AzureOpenAI:
    """Return an authenticated LLM client for the configured provider.

    The provider is read from config/config.yaml llm.provider.
    Switch providers by changing that value — no code changes needed.

    Returns:
        openai.OpenAI for provider='openai'
        openai.AzureOpenAI for provider='azure'

    Raises:
        ValueError: If provider is not supported.
        EnvironmentError: If required credentials are missing.
    """
    provider = _get_provider()
    if provider == "openai":
        return _build_openai_client()
    return _build_azure_client()


def get_model_name() -> str:
    """Return the model or deployment name for the configured provider.

    For OpenAI: returns the model name (e.g. 'gpt-4o').
    For Azure:  returns the deployment name from env or config.

    Returns:
        Model/deployment name string.

    Raises:
        EnvironmentError: If Azure deployment name cannot be resolved.
    """
    provider = _get_provider()

    if provider == "openai":
        return CFG["openai"]["model"]

    # Azure — deployment name from env (preferred) or config template
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if deployment:
        return deployment
    cfg_value = CFG.get("azure_openai", {}).get("chat_deployment", "")
    return _resolve_env_template(cfg_value)


def get_provider_cfg() -> dict[str, Any]:
    """Return the active provider config section.

    Callers use this for max_tokens, temperature, and other call-level
    parameters so they don't need to know which provider is active.

    Returns:
        Dict with at least: max_tokens (int), temperature (float).
    """
    provider = _get_provider()
    if provider == "openai":
        return CFG["openai"]
    return CFG["azure_openai"]
