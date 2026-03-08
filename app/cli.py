"""CLI entrypoint for the Google-Colab proof-of-concept app."""

from __future__ import annotations

import argparse
import json

from rich.console import Console
from rich.table import Table

from app.services import build_runtime_summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the Colab proof-of-concept app")
    parser.add_argument(
        "--name",
        default="Colab User",
        help="Name used in the greeting output.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write runtime metadata as JSON.",
    )
    return parser.parse_args()


def build_summary_table(summary: dict[str, str]) -> Table:
    """Create a rich table from runtime metadata."""
    table = Table(title="Colab Proof of Concept")
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Message", summary["message"])
    table.add_row("Timestamp (UTC)", summary["timestamp_utc"])
    table.add_row("Python", summary["python_version"])
    table.add_row("Platform", summary["platform"])
    return table


def run() -> None:
    """Run the application."""
    args = parse_args()
    summary = build_runtime_summary(args.name).to_dict()

    console = Console()
    console.print(build_summary_table(summary))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as json_file:
            json.dump(summary, json_file, ensure_ascii=False, indent=2)
        console.print(f"[green]Runtime metadata saved to {args.json_out}[/green]")
