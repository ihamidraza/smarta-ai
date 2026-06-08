#!/usr/bin/env python3
"""Interactive command-line chat with the LLM-backed agent.

The backend is chosen with the ``LLM_PROVIDER`` env var (default: local Ollama
running ``llama3.2:3b``). See ``.env.example`` and ``chatbot/providers.py``.

Usage:
    python main.py              # streaming chat (default)
    python main.py --no-stream  # wait for the full reply each turn

In-chat commands:
    /reset   clear the conversation history
    /save    save the conversation to sessions/<timestamp>.json
    /help    show commands
    /exit    quit  (also: /quit, Ctrl-D)
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from chatbot.agent import Agent, AgentError
from chatbot.config import ConfigError

console = Console()


def _banner(agent: Agent) -> str:
    return (
        f"[bold cyan]smarta-ai[/] — a smart agent powered by "
        f"[bold]{agent.config.provider}[/] ([italic]{agent.config.model}[/])\n"
        "Type your message, or [bold]/help[/] for commands. [bold]/exit[/] to quit."
    )


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        agent = Agent()
    except ConfigError as exc:
        console.print(f"[bold red]Configuration error:[/] {exc}")
        return 2

    console.print(Panel(_banner(agent), expand=False, border_style="cyan"))

    while True:
        try:
            user_input = console.input("[bold green]you ›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye![/]")
            return 0

        if not user_input:
            continue
        if user_input.startswith("/"):
            if _handle_command(user_input, agent):
                return 0
            continue

        try:
            _respond(agent, user_input, stream=not args.no_stream)
        except AgentError as exc:
            console.print(f"[bold red]Error:[/] {exc}")
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/]")


def _respond(agent: Agent, user_input: str, stream: bool) -> None:
    if stream:
        console.print("[bold magenta]ai ›[/] ", end="")
        for chunk in agent.stream(user_input):
            console.print(chunk, end="", soft_wrap=True, highlight=False, markup=False)
        console.print()  # newline after the streamed reply
    else:
        with console.status("[magenta]thinking…[/]", spinner="dots"):
            reply = agent.chat(user_input)
        console.print("[bold magenta]ai ›[/]")
        console.print(Markdown(reply))


def _handle_command(command: str, agent: Agent) -> bool:
    """Handle a /command. Returns True if the program should exit."""
    cmd = command.lower()
    if cmd in {"/exit", "/quit"}:
        console.print("[dim]bye![/]")
        return True
    if cmd == "/reset":
        agent.reset()
        console.print("[dim]conversation cleared.[/]")
    elif cmd == "/save":
        path = f"sessions/{dt.datetime.now():%Y%m%d-%H%M%S}.json"
        agent.conversation.save(path)
        console.print(f"[dim]saved to {path}[/]")
    elif cmd == "/help":
        console.print(
            "[dim]/reset  clear history\n"
            "/save   save conversation\n"
            "/help   this message\n"
            "/exit   quit[/]"
        )
    else:
        console.print(f"[yellow]Unknown command {command!r}. Try /help.[/]")
    return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-backed chat agent (provider set via LLM_PROVIDER)."
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="disable token streaming"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="log tool calls and info"
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
