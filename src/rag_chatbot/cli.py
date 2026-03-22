"""CLI: ingest PDFs and interactive async chat with Rich + loguru."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

from rag_chatbot.chain import build_rag_components, stream_generation_tokens
from rag_chatbot.config import load_settings
from rag_chatbot.ingest import run_ingest
from rag_chatbot.logging_setup import configure_logging

log = logger.bind(scope="cli")

# Throttle Live Markdown redraws while tokens stream (full re-parse each time).
_MARKDOWN_LIVE_INTERVAL_SEC = 0.08


def _answer_markdown_panel(text: str, *, streaming: bool) -> Panel:
    """Render assistant text as Rich Markdown (headings, lists, bold, fenced code)."""
    md = Markdown(
        text if text.strip() else " ",
        code_theme="monokai",
        inline_code_theme="monokai",
    )
    title = "[bold]Answer[/bold]"
    if streaming:
        title += " [dim](streaming…)[/dim]"
    return Panel(
        md,
        title=title,
        title_align="left",
        border_style="cyan" if streaming else "green",
        padding=(1, 2),
    )


async def _stream_answer_to_console(
    console: Console,
    generate_r,
    gen_input: dict,
    *,
    plain: bool,
) -> str:
    """Stream LLM tokens; either plain text or Live Markdown with throttled updates."""
    if plain:
        pieces: list[str] = []
        async for token in stream_generation_tokens(generate_r, gen_input):
            pieces.append(token)
            console.print(token, end="")
        console.print()
        return "".join(pieces)

    buffer = ""
    last_draw = time.monotonic()
    with Live(
        _answer_markdown_panel("", streaming=True),
        console=console,
        refresh_per_second=30,
        transient=False,
        vertical_overflow="visible",
    ) as live:
        async for token in stream_generation_tokens(generate_r, gen_input):
            buffer += token
            now = time.monotonic()
            if now - last_draw >= _MARKDOWN_LIVE_INTERVAL_SEC:
                live.update(_answer_markdown_panel(buffer, streaming=True))
                last_draw = now
        live.update(_answer_markdown_panel(buffer, streaming=False))

    console.print()
    return buffer


def _cmd_ingest(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)
    settings = load_settings()
    path = Path(args.path) if args.path else None
    console = Console(stderr=True)
    log.info("Starting ingest (clear_collection={})", not args.no_clear)
    try:
        n = run_ingest(settings, pdf_dir=path, clear_collection=not args.no_clear)
    except FileNotFoundError as e:
        log.error("{}", e)
        console.print(f"[red]{e}[/red]", file=sys.stderr)
        raise SystemExit(1) from e
    console.print(
        Panel.fit(
            f"[green]Indexed {n} chunks[/green]\n"
            f"Chroma: [cyan]{settings.chroma_path}[/cyan]\n"
            f"Collection: [cyan]{settings.chroma_collection}[/cyan]",
            title="Ingest complete",
            border_style="green",
        )
    )


async def _chat_loop(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)
    settings = load_settings()
    retrieve_r, generate_r, _full = build_rag_components(settings)

    console = Console()
    log.info(
        "Chat started (llm={}, embed={}, chroma={})",
        settings.ollama_llm_model,
        settings.ollama_embed_model,
        settings.chroma_path,
    )

    console.print(
        Panel.fit(
            "[bold]RAG PDF chatbot[/bold]\n"
            "• [cyan]Step 1[/cyan]: embed query + retrieve chunks from Chroma\n"
            "• [cyan]Step 2[/cyan]: stream answer from Ollama\n"
            "Type [dim]quit[/dim] or [dim]exit[/dim] to leave. Logs go to stderr.",
            title="Welcome",
            border_style="blue",
        )
    )

    while True:
        try:
            q = Prompt.ask("\n[bold cyan]Your question[/bold cyan]", console=console).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Goodbye.[/yellow]")
            log.info("Chat session ended (EOF/Ctrl+C)")
            break
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            log.info("Chat session ended (user quit)")
            console.print("[yellow]Goodbye.[/yellow]")
            break

        try:
            log.info("User question: {}", q[:200] + ("…" if len(q) > 200 else ""))

            with console.status(
                "[bold cyan]Step 1/2[/bold cyan] Embedding query and retrieving from Chroma…",
                spinner="dots",
            ):
                packed = await retrieve_r.ainvoke({"input": q})

            sources = packed.get("sources") or []
            tbl = Table(show_header=False, box=None, padding=(0, 1))
            tbl.add_column("Key", style="dim")
            tbl.add_column("Value")
            tbl.add_row("Retrieved chunks", str(packed.get("retrieved_chunk_count", "—")))
            tbl.add_row("Context size", f"{len(packed.get('context', ''))} chars")
            tbl.add_row("Source files", ", ".join(sources) if sources else "—")

            console.print(
                Panel(
                    tbl,
                    title="[bold]Step 1 done[/bold] — retrieval",
                    border_style="green",
                )
            )
            log.debug("Retrieval payload keys: {}", list(packed.keys()))

            gen_input = {"input": packed["input"], "context": packed["context"]}
            log.info("Step 2/2: streaming LLM response")

            console.print(
                Rule("[bold green]Step 2 — answer[/bold green]", style="green")
            )
            if args.plain:
                console.print(
                    "[dim]Plain text mode (no Markdown rendering).[/dim]\n"
                )
            else:
                console.print(
                    "[dim]Markdown mode: fenced code, lists, and emphasis render with Rich.[/dim]\n"
                )

            answer = await _stream_answer_to_console(
                console,
                generate_r,
                gen_input,
                plain=args.plain,
            )
            log.info("Answer length: {} chars", len(answer))

        except Exception as e:
            log.exception("Chat turn failed")
            console.print(f"[red]Error:[/red] {e}")


def _cmd_chat(args: argparse.Namespace) -> None:
    asyncio.run(_chat_loop(args))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rag_chatbot",
        description="RAG PDF QA with Ollama + Chroma (Rich CLI, loguru logs on stderr)",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Loguru level for stderr (default: INFO, or LOG_LEVEL env)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index PDFs into Chroma")
    p_ingest.add_argument(
        "--path",
        type=str,
        default=None,
        help="Directory containing PDF files (default: PDF_DIR / data/pdfs)",
    )
    p_ingest.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not drop the existing collection before indexing",
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    p_chat = sub.add_parser("chat", help="Interactive QA with Rich UI and streaming")
    p_chat.add_argument(
        "--plain",
        action="store_true",
        help="Stream raw text only (no Rich Markdown / code highlighting)",
    )
    p_chat.set_defaults(func=_cmd_chat)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
