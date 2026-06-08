"""A smart, tool-using chatbot agent backed by any OpenAI-compatible LLM.

Pick a backend with the ``LLM_PROVIDER`` env var (defaults to local Ollama).
See :mod:`chatbot.providers` for the built-in presets and how to add your own.
"""

from chatbot.agent import Agent
from chatbot.config import Config
from chatbot.providers import Provider

__all__ = ["Agent", "Config", "Provider"]
__version__ = "0.2.0"
