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
import socket
from dataclasses import dataclass
from typing import Any, Callable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx


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


# Weather is fetched from Open-Meteo (https://open-meteo.com): free, no API key.
# We first geocode the place name to coordinates, then ask for current weather.
_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_WEATHER_HTTP_TIMEOUT = 10.0

# WMO weather interpretation codes -> human-readable text.
_WEATHER_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def get_weather(location: str, units: str = "metric") -> str:
    """Return the current weather for a place name (e.g. 'Tokyo', 'Paris, France').

    ``units`` is 'metric' (°C, km/h) or 'imperial' (°F, mph).
    """
    location = (location or "").strip()
    if not location:
        return "Please provide a location, e.g. 'London' or 'Austin, Texas'."

    imperial = units.strip().lower() in {"imperial", "us", "f", "fahrenheit"}
    temp_unit = "°F" if imperial else "°C"
    wind_unit = "mph" if imperial else "km/h"

    try:
        with httpx.Client(timeout=_WEATHER_HTTP_TIMEOUT) as client:
            place = _geocode(client, location)
            if place is None:
                return f"Could not find a place called {location!r}. Try adding a country, e.g. 'Paris, France'."

            params = {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "weather_code,wind_speed_10m",
                "temperature_unit": "fahrenheit" if imperial else "celsius",
                "wind_speed_unit": "mph" if imperial else "kmh",
            }
            resp = client.get(_FORECAST_URL, params=params)
            resp.raise_for_status()
            current = resp.json().get("current", {})
    except httpx.TimeoutException:
        return "The weather service timed out. Please try again."
    except httpx.HTTPError as exc:
        return f"Could not reach the weather service: {exc}"

    code = current.get("weather_code")
    conditions = _WEATHER_CODES.get(code, f"weather code {code}")
    return (
        f"Current weather in {place['label']}: {conditions}, "
        f"{current.get('temperature_2m')}{temp_unit} "
        f"(feels like {current.get('apparent_temperature')}{temp_unit}), "
        f"humidity {current.get('relative_humidity_2m')}%, "
        f"wind {current.get('wind_speed_10m')} {wind_unit}."
    )


def _geocode(client: httpx.Client, location: str) -> dict[str, Any] | None:
    """Resolve a place name to coordinates plus a friendly label, or None.

    Open-Meteo's geocoder matches a single name, so "Austin, Texas" finds
    nothing. We try the full string first, then fall back to just the part
    before the first comma (the city).
    """
    candidates = [location]
    if "," in location:
        candidates.append(location.split(",", 1)[0].strip())

    results = None
    for name in candidates:
        resp = client.get(_GEOCODE_URL, params={"name": name, "count": 1})
        resp.raise_for_status()
        results = resp.json().get("results")
        if results:
            break
    if not results:
        return None
    top = results[0]
    label = ", ".join(
        part for part in (top.get("name"), top.get("admin1"), top.get("country")) if part
    )
    return {"latitude": top["latitude"], "longitude": top["longitude"], "label": label}


# Public IP comes from ipify (https://www.ipify.org): free, no API key.
_PUBLIC_IP_URL = "https://api.ipify.org"
_IP_HTTP_TIMEOUT = 10.0


def get_ip_address(kind: str = "both") -> str:
    """Report this machine's IP address.

    ``kind`` is 'local' (private LAN address), 'public' (internet-facing address),
    or 'both' (default).
    """
    kind = (kind or "both").strip().lower()
    if kind not in {"local", "public", "both"}:
        return "kind must be 'local', 'public', or 'both'."

    parts: list[str] = []
    if kind in {"local", "both"}:
        local = _local_ip()
        parts.append(
            f"Local (LAN) IP: {local}" if local else "Local (LAN) IP: unavailable"
        )
    if kind in {"public", "both"}:
        parts.append(_public_ip_line())
    return "\n".join(parts)


def _local_ip() -> str | None:
    """Best-effort private IP of the default network interface (no packets sent)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connecting a UDP socket doesn't send anything; it just picks the route
        # the OS would use to reach a public address, revealing our local IP.
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()


def _public_ip_line() -> str:
    try:
        with httpx.Client(timeout=_IP_HTTP_TIMEOUT) as client:
            resp = client.get(_PUBLIC_IP_URL, params={"format": "json"})
            resp.raise_for_status()
            return f"Public IP: {resp.json()['ip']}"
    except httpx.TimeoutException:
        return "Public IP: lookup timed out."
    except httpx.HTTPError as exc:
        return f"Public IP: could not be determined ({exc})."


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
    Tool(
        name="get_weather",
        description=(
            "Get the current weather (temperature, conditions, humidity, wind) "
            "for a city or place name."
        ),
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or place name, e.g. 'Tokyo' or 'Austin, Texas'.",
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "description": "'metric' for °C/km/h (default) or 'imperial' for °F/mph.",
                },
            },
            "required": ["location"],
        },
        func=get_weather,
    ),
    Tool(
        name="get_ip_address",
        description=(
            "Get the IP address of the machine running this assistant — the local "
            "LAN address, the public internet-facing address, or both."
        ),
        parameters={
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["local", "public", "both"],
                    "description": "Which address to report. Defaults to 'both'.",
                }
            },
            "required": [],
        },
        func=get_ip_address,
    ),
]

_BY_NAME: dict[str, Tool] = {t.name: t for t in TOOLS}


def schemas() -> list[dict[str, Any]]:
    """Return the OpenAI-format tool schemas for all registered tools."""
    return [t.to_openai_schema() for t in TOOLS]


def describe() -> str:
    """A short bulleted catalog of tools, for grounding the system prompt."""
    return "\n".join(f"- {t.name}: {t.description}" for t in TOOLS)


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
