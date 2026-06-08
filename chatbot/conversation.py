"""Conversation state with token-budgeted context management and compaction.

Context-engineering practices applied here:

* **Token budgeting** — history is bounded by an estimated *token* budget rather
  than a raw message count, which tracks real context-window usage far better.
* **Compaction over truncation** — when the budget is exceeded we *summarize* the
  oldest turns into a running summary instead of silently dropping them, so older
  context is condensed rather than lost.
* **Tool-output capping** — large tool results are truncated so a single noisy
  tool can't crowd out the rest of the conversation.
* **Tool-call integrity** — trimming never splits an assistant tool call from its
  results, nor starts the retained window on a dangling ``tool`` message.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

# A summarizer folds evicted messages into the running summary. It receives the
# previous summary (possibly "") and the list of evicted messages, and returns
# the new summary text. Kept as an injected callable so this module stays
# decoupled from the LLM client and remains easy to test.
Summarizer = Callable[[str, list[dict[str, Any]]], str]

# Rough, provider-agnostic token estimate. We cannot assume any specific
# tokenizer (Ollama, DeepSeek, OpenAI all differ), so ~4 characters per token is
# a deliberate, conservative heuristic for *budgeting*, not exact accounting.
_CHARS_PER_TOKEN = 4
_PER_MESSAGE_OVERHEAD_TOKENS = 4


class Conversation:
    """Holds the message history for a single chat session.

    Messages follow the OpenAI chat format: each is a dict with at least a
    ``role`` ("system" | "user" | "assistant" | "tool"). The system prompt is
    pinned first and never trimmed; a running summary of evicted turns (if any)
    is injected right after it.
    """

    def __init__(
        self,
        system_prompt: str,
        max_context_tokens: int = 6000,
        summarizer: Summarizer | None = None,
        max_tool_chars: int = 4000,
    ) -> None:
        self._system = {"role": "system", "content": system_prompt}
        self._messages: list[dict[str, Any]] = []
        self._summary: str = ""
        self._max_context_tokens = max_context_tokens
        self._summarizer = summarizer
        self._max_tool_chars = max_tool_chars

    # -- Mutation ------------------------------------------------------------

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})
        self._compact()

    def add_assistant(self, message: Any) -> None:
        """Append the assistant's reply (an SDK message object or a dict)."""
        if not isinstance(message, dict):
            message = message.model_dump(exclude_none=True)
        self._messages.append(message)
        self._compact()

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        # Cap oversized tool output so one noisy tool can't dominate the budget.
        if len(content) > self._max_tool_chars:
            content = content[: self._max_tool_chars] + "\n…[truncated]"
        self._messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )
        self._compact()

    def pop_last(self) -> None:
        """Remove the most recently appended message (used to roll back a turn)."""
        if self._messages:
            self._messages.pop()

    def reset(self) -> None:
        self._messages.clear()
        self._summary = ""

    # -- Rendering -----------------------------------------------------------

    def to_api_messages(self, dynamic_context: str | None = None) -> list[dict[str, Any]]:
        """Assemble the message list to send to the model.

        Order: pinned system prompt → fresh runtime context (date, tools, …) →
        running summary of older turns → live message history.
        """
        messages: list[dict[str, Any]] = [self._system]
        if dynamic_context:
            messages.append({"role": "system", "content": dynamic_context})
        if self._summary:
            messages.append(
                {
                    "role": "system",
                    "content": "Summary of earlier conversation (for context):\n"
                    + self._summary,
                }
            )
        messages.extend(self._messages)
        return messages

    # -- Compaction ----------------------------------------------------------

    def _compact(self) -> None:
        """Keep the context within budget, summarizing what we evict."""
        if self._tokens(self._stored()) <= self._max_context_tokens:
            return

        # Shrink the retained live history to ~half the budget, leaving headroom
        # so we don't summarize on every single turn.
        target = self._max_context_tokens // 2
        n = len(self._messages)
        # Always keep at least the two most recent messages intact.
        max_cut = max(0, n - 2)

        cut = 0
        while cut < max_cut:
            retained = self._messages[cut:]
            if self._tokens(self._with_overhead(retained)) <= target:
                break
            cut += 1

        # Never start the retained window on a dangling tool result; advance the
        # cut to keep each assistant tool-call grouped with its tool outputs.
        while cut < n and self._messages[cut]["role"] == "tool":
            cut += 1
        cut = min(cut, max_cut)

        if cut <= 0:
            return  # can't make progress (e.g. one huge recent message)

        evicted = self._messages[:cut]
        self._messages = self._messages[cut:]

        if self._summarizer and evicted:
            try:
                updated = self._summarizer(self._summary, evicted).strip()
                if updated:
                    self._summary = updated
            except Exception:  # noqa: BLE001 - summary is best-effort; drop on failure
                pass

    # -- Token estimation ----------------------------------------------------

    def _stored(self) -> list[dict[str, Any]]:
        """Everything that counts against the budget (system + summary + history)."""
        return self.to_api_messages()

    def _with_overhead(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The fixed parts (system + summary) plus the given messages."""
        fixed = [self._system]
        if self._summary:
            fixed.append({"role": "system", "content": self._summary})
        return [*fixed, *messages]

    @staticmethod
    def _tokens(messages: list[dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += _PER_MESSAGE_OVERHEAD_TOKENS
            content = msg.get("content") or ""
            total += len(content) // _CHARS_PER_TOKEN
            for call in msg.get("tool_calls") or []:
                total += len(json.dumps(call, default=str)) // _CHARS_PER_TOKEN
        return total

    # -- Persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "system": self._system,
            "summary": self._summary,
            "messages": self._messages,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._system = data.get("system", self._system)
        self._summary = data.get("summary", "")
        self._messages = data.get("messages", [])
