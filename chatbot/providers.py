"""Provider presets — how to reach each supported LLM backend.

The agent talks to any **OpenAI-compatible** chat-completions endpoint through
the official ``openai`` SDK. A :class:`Provider` is just the per-backend defaults
(base URL, default model, whether an API key is required) so that switching
providers is a one-line change in ``.env``:

    LLM_PROVIDER=ollama        # or: deepseek, openai, ...

To add a new provider, append a :class:`Provider` to :data:`PROVIDERS`. Anything
exposing an OpenAI-compatible ``/chat/completions`` API works out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Provider:
    """Connection defaults for one OpenAI-compatible LLM backend."""

    name: str
    base_url: str
    default_model: str
    # Hosted APIs need a real key; local servers like Ollama do not. When a key
    # is not required we still send ``placeholder_key`` because the SDK insists
    # on a non-empty value.
    api_key_required: bool = True
    placeholder_key: str = "not-needed"
    # Where users get a key / set the backend up — surfaced in error messages.
    help_url: str = ""


PROVIDERS: dict[str, Provider] = {
    # Local, offline-friendly default. Run `ollama serve` and
    # `ollama pull llama3.2:3b` first. Ollama exposes an OpenAI-compatible API
    # at /v1, so the same code path works unchanged.
    "ollama": Provider(
        name="ollama",
        base_url="http://localhost:11434/v1",
        default_model="llama3.2:3b",
        api_key_required=False,
        placeholder_key="ollama",
        help_url="https://ollama.com/download",
    ),
    "deepseek": Provider(
        name="deepseek",
        base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        api_key_required=True,
        help_url="https://platform.deepseek.com/",
    ),
    "openai": Provider(
        name="openai",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        api_key_required=True,
        help_url="https://platform.openai.com/api-keys",
    ),
}

DEFAULT_PROVIDER = "ollama"


def get_provider(name: str) -> Provider | None:
    """Look up a provider preset by name (case-insensitive)."""
    return PROVIDERS.get(name.strip().lower())


def available() -> list[str]:
    """Names of all built-in providers, for error messages and help text."""
    return sorted(PROVIDERS)
