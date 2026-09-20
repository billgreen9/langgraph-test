"""jieba 分词，全文召回与 BM25 / 覆盖率共用。"""

from __future__ import annotations

import logging
import re

import jieba

jieba.setLogLevel(logging.WARNING)

_PUNCT = re.compile(r"^[\W_]+$", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    text = (text or "").strip().lower()
    if not text:
        return []
    out: list[str] = []
    for word in jieba.lcut(text):
        word = word.strip()
        if not word or _PUNCT.fullmatch(word):
            continue
        out.append(word)
    return out


def tokens_for_tsquery(text: str) -> str:
    parts = [t.replace("'", "").replace(":", "") for t in tokenize(text)]
    parts = [t for t in parts if t]
    return " | ".join(parts)


def tokens_for_tsvector(text: str) -> str:
    return " ".join(tokenize(text))


def token_overlap(query: str, doc: str) -> float:
    q = set(tokenize(query))
    if not q:
        return 0.0
    return len(q & set(tokenize(doc))) / len(q)
