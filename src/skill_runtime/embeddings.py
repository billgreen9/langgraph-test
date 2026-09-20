"""文档 / query 向量。接口不可用时跳过语义召回。"""

from __future__ import annotations

import logging

from ..config import settings

logger = logging.getLogger(__name__)

_embeddings = None
_failed = False


def _client():
    global _embeddings, _failed
    if _failed:
        return None
    if _embeddings is not None:
        return _embeddings
    if not settings.openai_api_key:
        _failed = True
        return None
    try:
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict = {
            "model": settings.embedding_model,
            "api_key": settings.openai_api_key,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        _embeddings = OpenAIEmbeddings(**kwargs)
        return _embeddings
    except Exception:
        logger.exception("初始化 embedding 失败")
        _failed = True
        return None


def embed_texts(texts: list[str]) -> list[list[float] | None]:
    if not texts:
        return []
    client = _client()
    if client is None:
        return [None] * len(texts)
    try:
        return [list(map(float, v)) for v in client.embed_documents(texts)]
    except Exception:
        logger.exception("批量 embedding 失败")
        return [None] * len(texts)


def embed_query(text: str) -> list[float] | None:
    client = _client()
    if client is None or not (text or "").strip():
        return None
    try:
        return [float(x) for x in client.embed_query(text)]
    except Exception:
        logger.exception("query embedding 失败")
        return None
