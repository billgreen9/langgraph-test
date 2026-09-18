"""Entry point for the hello-langgraph project."""

from __future__ import annotations

import asyncio
import sys

from .graph import run_graph


async def main() -> None:
    """Ask a sample question and print the graph's answer."""
    question = " ".join(sys.argv[1:]) or "What is LangGraph in one sentence?"
    print(f"Q: {question}")
    answer = await run_graph(question)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
