"""单独跑长任务图。图节点会请求 API 的 ``GET /slow``。

先启动接口，再跑本程序::

    uvicorn src.api.main:app --port 8000
    python -m src.long_task.main
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from .graph import DEFAULT_SLOW_URL, build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 LangGraph 长任务节点")
    parser.add_argument(
        "--url",
        default=DEFAULT_SLOW_URL,
        help=f"长任务请求的接口，默认 {DEFAULT_SLOW_URL}",
    )
    return parser.parse_args()


async def run(url: str) -> dict:
    app = build_graph()
    return await app.ainvoke({"url": url, "status": "running"})


def main() -> None:
    args = parse_args()
    result = asyncio.run(run(args.url))
    print(result)


if __name__ == "__main__":
    main()
