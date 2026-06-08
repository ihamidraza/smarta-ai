# chat-ai

A smart, tool-using chat **agent** in Python that works with **any
OpenAI-compatible LLM** — local models via [Ollama](https://ollama.com/),
[DeepSeek](https://platform.deepseek.com/), [OpenAI](https://platform.openai.com/),
and more. It ships configured for a **local Ollama** model (`llama3.2:3b`) so you
can run it offline with no API key.

It is a real agent, not just a wrapper: it keeps multi-turn memory, streams replies token-by-token, and can call tools (functions) to answer questions it can't answer from the model alone — e.g. the current time or arithmetic.

## Features

- 🔌 **Provider-agnostic** — switch backends with one env var. Local or hosted.
- 🧠 **Multi-turn memory** with automatic context trimming so long chats stay cheap.
- 🔧 **Tool calling** (OpenAI-compatible function calling) — easily add your own tools.
- ⚡ **Streaming** responses for a responsive feel.
- ⚙️ **Config via environment / `.env`** — no secrets in code.
- 💾 **Save / reset** sessions from inside the chat.
- 🧱 Clean, typed, modular code following Python best practices.

## Project layout

```
chat-ai/
├── main.py                 # CLI entry point
├── chatbot/
│   ├── config.py           # env-based configuration
│   ├── providers.py        # LLM backend presets (ollama, deepseek, openai, ...)
│   ├── agent.py            # the agent: API calls + tool loop
│   ├── conversation.py     # message history management
│   └── tools.py            # the tools the agent can call
├── requirements.txt
└── .env.example
```

## Setup

```bash
# 1. (recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. configure
cp .env.example .env        # defaults to local Ollama — no key needed
```

### Default: local Ollama (no API key)

Install Ollama from <https://ollama.com/download>, then pull the model:

```bash
ollama serve                # start the local server (if not already running)
ollama pull llama3.2:3b     # download the model
```

That's it — the bundled `.env` already points at Ollama with `llama3.2:3b`.

## Usage

```bash
python main.py                # streaming chat
python main.py --no-stream    # render full replies as Markdown
python main.py -v             # also log tool calls
```

In-chat commands: `/reset`, `/save`, `/help`, `/exit`.

### Example

```
you › what's 2^10 * 3?
ai › That's 3072.    (the agent called the `calculate` tool)

you › and what time is it in Tokyo?
ai › It's Sunday, 07 June 2026, 22:14:03 JST.   (called `get_current_time`)
```

## Choosing a different model or provider

Everything is driven by env vars (set them in `.env` or your shell):

| Variable | Purpose | Example |
| --- | --- | --- |
| `LLM_PROVIDER` | Which backend preset to use | `ollama`, `deepseek`, `openai` |
| `LLM_MODEL` | Model name (defaults per provider) | `llama3.2:3b`, `deepseek-chat`, `gpt-4o-mini` |
| `LLM_API_KEY` | API key (not needed for Ollama) | `sk-...` |
| `LLM_BASE_URL` | Override the endpoint | `http://localhost:11434/v1` |
| `LLM_TEMPERATURE` | Sampling temperature | `0.7` |
| `LLM_MAX_TOKENS` | Max tokens per reply | `2048` |
| `LLM_TIMEOUT` | Request timeout (seconds) | `60` |

A few common setups:

```bash
# Another local Ollama model
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b

# DeepSeek (hosted)
LLM_PROVIDER=deepseek
LLM_API_KEY=sk-...

# OpenAI (hosted)
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

### Adding a new provider

Any backend exposing an OpenAI-compatible `/chat/completions` API works. Add a
`Provider(...)` entry to `PROVIDERS` in `chatbot/providers.py` (base URL, default
model, whether a key is required) — or just set `LLM_BASE_URL` directly for a
one-off endpoint.

## Adding a tool

Open `chatbot/tools.py`, write a function that returns a string, and append a
`Tool(...)` entry to `TOOLS` describing its parameters. That's it — the agent
picks it up automatically.

## Using the agent in your own code

```python
from chatbot import Agent

agent = Agent()                       # reads config from env / .env
print(agent.chat("Hello!"))           # blocking, returns the full reply
for chunk in agent.stream("Tell me a joke"):
    print(chunk, end="")
```

## Notes

This project uses the official `openai` SDK pointed at whichever provider's
base URL you select. OpenAI-compatible tool calling requires a model that
supports it — `llama3.2:3b` and most modern models do.
