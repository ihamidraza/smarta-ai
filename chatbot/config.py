"""Centralized configuration loaded from the environment.

All runtime knobs live here so the rest of the code never reads ``os.environ``
directly. Values come from real environment variables or a local ``.env`` file
(see ``.env.example``).

The agent is provider-agnostic: pick a backend with ``LLM_PROVIDER`` (e.g.
``ollama``, ``deepseek``, ``openai``) and the matching defaults from
:mod:`chatbot.providers` fill in. Any ``LLM_*`` variable overrides those
defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from chatbot.providers import DEFAULT_PROVIDER, available, get_provider

# Load `.env` once, at import time. Real environment variables always win.
load_dotenv(override=False)


class ConfigError(RuntimeError):
    """Raised when the configuration is missing or invalid."""


# Defaults live here as module constants (not just dataclass field defaults) so
# ``from_env`` can reference them. With ``slots=True`` the field defaults are not
# readable as ``Config.<field>`` — that returns the slot descriptor, not the value.
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_MAX_TOKENS = 2048
_DEFAULT_TIMEOUT = 60.0
# Input-side context budget (estimated tokens). When history exceeds this, the
# oldest turns are summarized into a running summary instead of being dropped.
_DEFAULT_MAX_CONTEXT_TOKENS = 6000
_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, friendly, and knowledgeable assistant. "
    "Answer clearly and concisely. When you use a tool, explain the result "
    "in plain language. If you are unsure about something, say so."
)


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable snapshot of the agent's configuration."""

    provider: str
    api_key: str
    base_url: str
    model: str
    temperature: float = _DEFAULT_TEMPERATURE
    max_tokens: int = _DEFAULT_MAX_TOKENS
    request_timeout: float = _DEFAULT_TIMEOUT
    max_context_tokens: int = _DEFAULT_MAX_CONTEXT_TOKENS
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from environment variables, validating as we go."""
        provider_name = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
        provider = get_provider(provider_name)
        if provider is None:
            raise ConfigError(
                f"Unknown LLM_PROVIDER {provider_name!r}. "
                f"Built-in providers: {', '.join(available())}. "
                "Add your own in chatbot/providers.py, or set LLM_BASE_URL to any "
                "OpenAI-compatible endpoint."
            )

        # API key: required for hosted providers, optional for local ones.
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            if provider.api_key_required:
                hint = f" Get one at {provider.help_url}." if provider.help_url else ""
                raise ConfigError(
                    f"LLM_API_KEY is not set, but provider {provider.name!r} requires "
                    f"a key.{hint} Set it in .env or export LLM_API_KEY in your shell."
                )
            api_key = provider.placeholder_key

        return cls(
            provider=provider.name,
            api_key=api_key,
            base_url=os.getenv("LLM_BASE_URL", provider.base_url).strip(),
            model=os.getenv("LLM_MODEL", provider.default_model).strip(),
            temperature=_get_float("LLM_TEMPERATURE", _DEFAULT_TEMPERATURE),
            max_tokens=_get_int("LLM_MAX_TOKENS", _DEFAULT_MAX_TOKENS),
            request_timeout=_get_float("LLM_TIMEOUT", _DEFAULT_TIMEOUT),
            max_context_tokens=_get_int(
                "LLM_MAX_CONTEXT_TOKENS", _DEFAULT_MAX_CONTEXT_TOKENS
            ),
        )


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
