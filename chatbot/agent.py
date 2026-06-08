"""The Agent: orchestrates the LLM client, history, and tool calling.

The client is the OpenAI SDK pointed at whichever OpenAI-compatible backend the
:class:`~chatbot.config.Config` selects (Ollama, DeepSeek, OpenAI, ...), so the
same code path serves every provider.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections.abc import Iterator
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    OpenAIError,
)

from chatbot import tools
from chatbot.config import Config
from chatbot.conversation import Conversation

logger = logging.getLogger(__name__)

# Hard cap on tool-calling rounds per user turn, to avoid an infinite loop if
# the model keeps requesting tools.
_MAX_TOOL_ROUNDS = 5

# Retry transient network failures with exponential backoff.
_MAX_RETRIES = 2
_RETRY_BACKOFF = 1.5


class Agent:
    """A multi-turn, tool-using chat agent backed by any OpenAI-compatible LLM."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.request_timeout,
        )
        self.conversation = Conversation(
            self.config.system_prompt,
            max_context_tokens=self.config.max_context_tokens,
            summarizer=self._summarize,
        )

    # -- Public API ----------------------------------------------------------

    def chat(self, user_input: str) -> str:
        """Send one user turn and return the assistant's final text reply.

        Handles any tool calls the model makes along the way before returning.
        """
        self.conversation.add_user(user_input)

        for _ in range(_MAX_TOOL_ROUNDS):
            message = self._complete()
            self.conversation.add_assistant(message)

            if not message.tool_calls:
                return message.content or ""

            self._run_tool_calls(message.tool_calls)

        # Ran out of tool rounds; ask once more for a plain answer.
        final = self._complete(use_tools=False)
        self.conversation.add_assistant(final)
        return final.content or ""

    def stream(self, user_input: str) -> Iterator[str]:
        """Stream the assistant's reply token-by-token.

        Note: streaming and tool calls don't mix cleanly, so if the model wants
        a tool we fall back to the non-streaming :meth:`chat` path and yield the
        finished reply in one chunk.
        """
        self.conversation.add_user(user_input)
        try:
            stream = self._client.chat.completions.create(
                model=self.config.model,
                messages=self.conversation.to_api_messages(self._dynamic_context()),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                tools=tools.schemas(),
                stream=True,
            )
        except OpenAIError as exc:
            raise AgentError(_friendly_error(exc, self.config)) from exc

        chunks: list[str] = []
        wants_tool = False
        for event in stream:
            delta = event.choices[0].delta
            if delta.tool_calls:
                wants_tool = True
                break
            if delta.content:
                chunks.append(delta.content)
                yield delta.content

        if wants_tool:
            # Roll back the streamed user turn already recorded, then re-run the
            # full tool-aware path so tool calls are handled correctly.
            self.conversation.pop_last()  # remove the user msg we just added
            yield self.chat(user_input)
            return

        self.conversation.add_assistant({"role": "assistant", "content": "".join(chunks)})

    def reset(self) -> None:
        """Clear history (keeps the system prompt)."""
        self.conversation.reset()

    # -- Context engineering -------------------------------------------------

    def _dynamic_context(self) -> str:
        """Fresh runtime context injected as a system message each turn.

        Grounds the model in the current date/time and its actual tool catalog so
        it neither hallucinates "today" nor calls a tool when a direct answer will
        do. Built per turn so the date never goes stale within a session.
        """
        now = dt.datetime.now().astimezone()
        return (
            f"Current date and time: {now:%A, %d %B %Y, %H:%M %Z}.\n"
            f"Backend: provider '{self.config.provider}', model "
            f"'{self.config.model}'.\n"
            "Available tools:\n"
            f"{tools.describe()}\n"
            "Tool-use policy:\n"
            "- Call `web_search` whenever the question is about specific facts, "
            "people, places, organizations, or topics you are not confident about "
            "or that may be outside your training. When unsure, search rather than "
            "guess.\n"
            "- Use `get_weather`, `get_ip_address`, `get_current_time`, and "
            "`calculate` for that live or computed data.\n"
            "- Answer directly only for general knowledge, reasoning, or chit-chat "
            "that doesn't need fresh data.\n"
            "Base answers on tool results when you use them, cite the source URLs "
            "from `web_search`, and never invent tool output or links."
        )

    def _summarize(self, previous_summary: str, evicted: list[dict[str, Any]]) -> str:
        """Fold evicted messages into a concise running summary (compaction).

        Used as the :class:`Conversation` summarizer. Runs as a standalone
        completion (tools disabled, low temperature) and does not touch the live
        conversation, so it cannot recurse into further compaction.
        """
        instruction = (
            "You compact a chat transcript into a concise running summary that "
            "preserves continuity. Keep facts, decisions, named entities, numbers, "
            "and any unresolved questions or user preferences. Drop pleasantries and "
            "filler. Return only the updated summary, a short paragraph or bullets."
        )
        request = (
            f"Existing summary:\n{previous_summary or '(none yet)'}\n\n"
            f"New messages to fold in:\n{_render_transcript(evicted)}\n\n"
            "Return the updated summary only."
        )
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": request},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content or previous_summary

    # -- Internals -----------------------------------------------------------

    def _complete(self, use_tools: bool = True) -> Any:
        """One non-streaming completion; returns the assistant message object."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": self.conversation.to_api_messages(self._dynamic_context()),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if use_tools:
            kwargs["tools"] = tools.schemas()

        last_exc: OpenAIError | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message
            except (APIConnectionError, APITimeoutError) as exc:
                # Transient: worth retrying with backoff.
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BACKOFF**attempt
                    logger.warning("network error (attempt %d), retrying in %.1fs", attempt + 1, delay)
                    time.sleep(delay)
                    continue
            except OpenAIError as exc:
                # Non-transient (auth, balance, bad request): fail fast.
                raise AgentError(_friendly_error(exc, self.config)) from exc

        raise AgentError(_friendly_error(last_exc, self.config)) from last_exc

    def _run_tool_calls(self, tool_calls: list[Any]) -> None:
        for call in tool_calls:
            name = call.function.name
            args = call.function.arguments or "{}"
            logger.info("tool call: %s(%s)", name, args)
            result = tools.dispatch(name, args)
            self.conversation.add_tool_result(call.id, result)


def _render_transcript(messages: list[dict[str, Any]]) -> str:
    """Flatten message dicts into a readable transcript for summarization."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        content = (msg.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            lines.append(f"{role} called tool {fn.get('name')}({fn.get('arguments')})")
    return "\n".join(lines) if lines else "(no textual content)"


class AgentError(RuntimeError):
    """User-facing error raised when the agent cannot complete a request."""


def _friendly_error(exc: OpenAIError | None, config: Config) -> str:
    """Map an SDK error to an actionable, human-readable message."""
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        hint = (
            "Is `ollama serve` running and the model pulled "
            f"(`ollama pull {config.model}`)? "
            if config.provider == "ollama"
            else "Check your internet connection and any proxy/VPN. "
        )
        return (
            f"Could not reach the {config.provider} backend at {config.base_url} "
            f"(network/timeout). {hint}"
            "You can also override the endpoint with LLM_BASE_URL."
        )
    if isinstance(exc, APIStatusError):
        status = exc.status_code
        # Pull the server's message when present for a precise reason.
        server_msg = ""
        try:
            server_msg = exc.response.json().get("error", {}).get("message", "")
        except Exception:  # noqa: BLE001 - body may be empty or non-JSON
            pass
        reasons = {
            401: "Authentication failed — check that LLM_API_KEY is correct.",
            402: (
                "Insufficient balance / quota on your account with provider "
                f"{config.provider!r}. Top up or switch LLM_PROVIDER."
            ),
            404: (
                f"Model {config.model!r} not found on provider {config.provider!r}. "
                "Check LLM_MODEL"
                + (
                    f" (pull it with `ollama pull {config.model}`)."
                    if config.provider == "ollama"
                    else "."
                )
            ),
            429: "Rate limited by the API. Wait a moment and try again.",
        }
        base = reasons.get(status, f"API returned HTTP {status}.")
        return f"{base} ({server_msg})" if server_msg and status not in reasons else base
    return f"API request failed: {exc}"
