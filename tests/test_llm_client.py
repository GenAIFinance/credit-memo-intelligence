"""
tests/test_llm_client.py

Unit tests for src/utils/llm_client.py.
Covers Codex review findings:
  - provider validation rejects unsupported values
  - ${VAR} template resolution errors clearly when env var missing
  - OpenAI vs Azure client builder selection path
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cfg(provider: str) -> dict:
    """Build a minimal CFG dict for a given provider."""
    return {
        "llm": {"provider": provider},
        "openai": {
            "api_key": "${OPENAI_API_KEY}",
            "model": "gpt-4o",
            "max_tokens": 1000,
            "temperature": 0.5,
        },
        "azure_openai": {
            "endpoint": "${AZURE_OPENAI_ENDPOINT}",
            "api_key": "${AZURE_OPENAI_API_KEY}",
            "api_version": "2024-02-01",
            "chat_deployment": "${AZURE_OPENAI_DEPLOYMENT}",
            "max_tokens": 1000,
            "temperature": 0.5,
        },
    }


# ── Provider validation ───────────────────────────────────────────────────────

class TestGetProvider:
    def test_openai_is_valid(self):
        from src.utils.llm_client import _get_provider
        with patch("src.utils.llm_client.CFG", _cfg("openai")):
            assert _get_provider() == "openai"

    def test_azure_is_valid(self):
        from src.utils.llm_client import _get_provider
        with patch("src.utils.llm_client.CFG", _cfg("azure")):
            assert _get_provider() == "azure"

    def test_unsupported_provider_raises(self):
        from src.utils.llm_client import _get_provider
        with patch("src.utils.llm_client.CFG", _cfg("anthropic")):
            with pytest.raises(ValueError, match="Unsupported LLM provider"):
                _get_provider()

    def test_empty_provider_raises(self):
        from src.utils.llm_client import _get_provider
        with patch("src.utils.llm_client.CFG", _cfg("")):
            with pytest.raises(ValueError, match="Unsupported LLM provider"):
                _get_provider()


# ── Env template resolution ───────────────────────────────────────────────────

class TestResolveEnvTemplate:
    def test_literal_value_passes_through(self):
        from src.utils.llm_client import _resolve_env_template
        assert _resolve_env_template("gpt-4o") == "gpt-4o"

    def test_template_resolves_from_env(self):
        from src.utils.llm_client import _resolve_env_template
        with patch.dict(os.environ, {"MY_KEY": "resolved-value"}):
            assert _resolve_env_template("${MY_KEY}") == "resolved-value"

    def test_missing_env_var_raises_clearly(self):
        from src.utils.llm_client import _resolve_env_template
        with patch.dict(os.environ, {}, clear=True):
            # Remove the var if it exists
            os.environ.pop("MISSING_VAR", None)
            with pytest.raises(EnvironmentError, match="MISSING_VAR"):
                _resolve_env_template("${MISSING_VAR}")

    def test_partial_template_not_resolved(self):
        """Values that aren't exactly ${VAR} are treated as literals."""
        from src.utils.llm_client import _resolve_env_template
        assert _resolve_env_template("prefix-${VAR}") == "prefix-${VAR}"

    def test_whitespace_around_template_handled(self):
        from src.utils.llm_client import _resolve_env_template
        with patch.dict(os.environ, {"MY_KEY": "val"}):
            assert _resolve_env_template("  ${MY_KEY}  ") == "val"


# ── Client builder selection ──────────────────────────────────────────────────

class TestGetLlmClient:
    def test_openai_provider_builds_openai_client(self):
        from src.utils.llm_client import get_llm_client
        mock_client = MagicMock()
        with patch("src.utils.llm_client.CFG", _cfg("openai")), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}), \
             patch("src.utils.llm_client.OpenAI", return_value=mock_client) as mock_cls:
            client = get_llm_client()
            mock_cls.assert_called_once()
            assert client is mock_client

    def test_azure_provider_builds_azure_client(self):
        from src.utils.llm_client import get_llm_client
        mock_client = MagicMock()
        with patch("src.utils.llm_client.CFG", _cfg("azure")), \
             patch.dict(os.environ, {
                 "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
                 "AZURE_OPENAI_API_KEY": "test-key",
             }), \
             patch("src.utils.llm_client.AzureOpenAI", return_value=mock_client) as mock_cls:
            client = get_llm_client()
            mock_cls.assert_called_once()
            assert client is mock_client

    def test_openai_missing_key_raises(self):
        from src.utils.llm_client import get_llm_client
        with patch("src.utils.llm_client.CFG", _cfg("openai")), \
             patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
                get_llm_client()

    def test_azure_missing_credentials_raises(self):
        from src.utils.llm_client import get_llm_client
        with patch("src.utils.llm_client.CFG", _cfg("azure")), \
             patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AZURE_OPENAI_ENDPOINT", None)
            os.environ.pop("AZURE_OPENAI_API_KEY", None)
            with pytest.raises(EnvironmentError):
                get_llm_client()


# ── get_model_name ────────────────────────────────────────────────────────────

class TestGetModelName:
    def test_openai_returns_model_from_config(self):
        from src.utils.llm_client import get_model_name
        with patch("src.utils.llm_client.CFG", _cfg("openai")):
            assert get_model_name() == "gpt-4o"

    def test_azure_returns_deployment_from_env(self):
        from src.utils.llm_client import get_model_name
        with patch("src.utils.llm_client.CFG", _cfg("azure")), \
             patch.dict(os.environ, {"AZURE_OPENAI_DEPLOYMENT": "my-gpt4o"}):
            assert get_model_name() == "my-gpt4o"


# ── get_provider_cfg ──────────────────────────────────────────────────────────

class TestGetProviderCfg:
    def test_openai_returns_openai_section(self):
        from src.utils.llm_client import get_provider_cfg
        cfg = _cfg("openai")
        with patch("src.utils.llm_client.CFG", cfg):
            result = get_provider_cfg()
            assert result["model"] == "gpt-4o"
            assert "max_tokens" in result

    def test_azure_returns_azure_section(self):
        from src.utils.llm_client import get_provider_cfg
        cfg = _cfg("azure")
        with patch("src.utils.llm_client.CFG", cfg):
            result = get_provider_cfg()
            assert "chat_deployment" in result
            assert "api_version" in result
