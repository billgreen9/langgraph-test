"""简单的 HTTP API 服务。

PyCharm 调试：直接 Debug 本文件（不要点 ``app = FastAPI()`` 旁边的运行按钮）。
命令行::

    python -m src.api.main
    uvicorn src.api.main:app --reload --port 8000

接口：
- ``GET  /``        服务信息
- ``GET  /health``  健康检查
- ``GET  /slow``    模拟耗时超过 5 分钟的处理
- ``POST /echo``    回显请求体中的 message
"""

from __future__ import annotations

import asyncio
import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="langgraph-test API", version="0.1.0")


class EchoRequest(BaseModel):
    message: str = Field(..., min_length=1, description="要回显的文本")


class EchoResponse(BaseModel):
    message: str


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "langgraph-test-api", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/slow")
async def slow() -> dict[str, float | str]:
    """模拟一次耗时超过 5 分钟的处理，请求会一直挂起直到结束。"""
    started = time.monotonic()
    await asyncio.sleep(5 * 60 + 5)
    elapsed = time.monotonic() - started
    return {"status": "done", "elapsed_seconds": round(elapsed, 1)}


@app.post("/echo", response_model=EchoResponse)
def echo(body: EchoRequest) -> EchoResponse:
    return EchoResponse(message=body.message)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
