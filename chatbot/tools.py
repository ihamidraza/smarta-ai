"""Tools the agent can call.

Each tool is a plain Python function plus a JSON-schema description that the LLM
sees. The agent uses the OpenAI tool-calling format (supported by Ollama,
DeepSeek, OpenAI, and other compatible backends), so registering a tool is just:
write the function, describe it, add it to ``TOOLS``.

To add your own tool:
    1. Write a function that takes JSON-serializable kwargs and returns a string.
    2. Append a Tool(...) entry to TOOLS describing its parameters.
"""

from __future__ import annotations

import ast
import datetime as _dt
import json
import operator as _op
from dataclasses import dataclass
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True, slots=True)
class Tool:
    """A callable tool plus the JSON schema advertised to the model."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# --- Tool implementations ---------------------------------------------------


def get_current_time(timezone: str = "UTC") -> str:
    """Return the current date and time in the given IANA timezone."""
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return f"Unknown timezone {timezone!r}. Use an IANA name like 'America/New_York'."
    now = _dt.datetime.now(tz)
    return now.strftime("%A, %d %B %Y, %H:%M:%S %Z")


# A tiny, safe arithmetic evaluator. We deliberately do NOT use eval().
_ALLOWED_BINOPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: _op.pos, ast.USub: _op.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression (+ - * / // % ** and parentheses)."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return f"Could not evaluate {expression!r}: {exc}"
    return f"{expression} = {result}"


# --- Registry ---------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="get_current_time",
        description="Get the current date and time in a given IANA timezone.",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone, e.g. 'UTC', 'America/New_York'.",
                }
            },
            "required": [],
        },
        func=get_current_time,
    ),
    Tool(
        name="calculate",
        description="Evaluate a basic arithmetic expression and return the result.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '(2 + 3) * 4'.",
                }
            },
            "required": ["expression"],
        },
        func=calculate,
    ),
]

_BY_NAME: dict[str, Tool] = {t.name: t for t in TOOLS}


def schemas() -> list[dict[str, Any]]:
    """Return the OpenAI-format tool schemas for all registered tools."""
    return [t.to_openai_schema() for t in TOOLS]


def dispatch(name: str, arguments: str) -> str:
    """Run a tool by name with JSON-encoded ``arguments`` from the model."""
    tool = _BY_NAME.get(name)
    if tool is None:
        return f"Error: unknown tool {name!r}."
    try:
        kwargs = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError as exc:
        return f"Error: could not parse arguments for {name!r}: {exc}"
    try:
        return tool.func(**kwargs)
    except TypeError as exc:
        return f"Error: bad arguments for {name!r}: {exc}"
    except Exception as exc:  # noqa: BLE001 - report any tool failure to the model
        return f"Error while running {name!r}: {exc}"
