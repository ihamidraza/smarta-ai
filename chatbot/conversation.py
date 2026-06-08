"""Conversation state: the running message history sent to the model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Conversation:
    """Holds the message history for a single chat session.

    Messages follow the OpenAI chat format: each is a dict with at least
    a ``role`` ("system" | "user" | "assistant" | "tool"). The system prompt is
    pinned as the first message and is never trimmed.
    """

    def __init__(self, system_prompt: str, max_messages: int = 40) -> None:
        # max_messages bounds how many non-system messages we keep, so a long
        # session does not grow the context (and cost) without limit.
        self._system = {"role": "system", "content": system_prompt}
        self._messages: list[dict[str, Any]] = []
        self._max_messages = max_messages

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, message: Any) -> None:
        """Append the assistant's reply (an SDK message object or a dict)."""
        if not isinstance(message, dict):
            message = message.model_dump(exclude_none=True)
        self._messages.append(message)
        self._trim()

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self._messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        )

    def to_api_messages(self) -> list[dict[str, Any]]:
        """Full message list to send to the API, system prompt first."""
        return [self._system, *self._messages]

    def reset(self) -> None:
        self._messages.clear()

    def _trim(self) -> None:
        if len(self._messages) <= self._max_messages:
            return
        # Drop the oldest messages, but never start the window on a dangling
        # "tool" message (which must follow its assistant tool_call).
        overflow = len(self._messages) - self._max_messages
        while overflow < len(self._messages) and self._messages[overflow]["role"] == "tool":
            overflow += 1
        self._messages = self._messages[overflow:]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"system": self._system, "messages": self._messages}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._system = data.get("system", self._system)
        self._messages = data.get("messages", [])
