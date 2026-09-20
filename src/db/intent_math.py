"""intent_math：query 积累、全文召回、向量召回。"""

from __future__ import annotations

from typing import Any

from .base import TableRepo
from .models import IntentMathRow, utcnow


def _embedding_literal(vec: list[float] | None) -> str | None:
    if not vec:
        return None
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def _parse_embedding(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [float(x) for x in inner.split(",")]
    return None


def row_to_intent(row: dict[str, Any]) -> IntentMathRow:
    return IntentMathRow(
        id=row.get("id"),
        msg=row["msg"],
        score_limit=float(row.get("score_limit") or 0),
        skill_id=row.get("skill_id"),
        skill_level=int(row.get("skill_level") or 0),
        answer=row.get("answer") or "",
        forbid=int(row.get("forbid") or 0),
        source=row.get("source") or "seed",
        seed=row.get("seed") or "",
        enabled=bool(row.get("enabled", True)),
        msg_embedding=_parse_embedding(row.get("msg_embedding")),
    )


class IntentMathRepo(TableRepo):
    def upsert(self, row: IntentMathRow, tokens: str) -> None:
        sql = """
            INSERT INTO intent_math
                (msg, msg_tsv, msg_embedding, score_limit, skill_id, skill_key,
                 skill_level, answer, forbid, source, seed, enabled, updated_at)
            VALUES (
                %(msg)s, to_tsvector('simple', %(tokens)s),
                %(emb)s::vector, %(score_limit)s, %(skill_id)s, %(skill_key)s,
                %(skill_level)s, %(answer)s, %(forbid)s, %(source)s, %(seed)s,
                %(enabled)s, %(ts)s
            )
            ON CONFLICT (msg, skill_key) DO UPDATE SET
                msg_tsv = EXCLUDED.msg_tsv,
                score_limit = EXCLUDED.score_limit,
                skill_id = EXCLUDED.skill_id,
                skill_level = EXCLUDED.skill_level,
                answer = EXCLUDED.answer,
                forbid = EXCLUDED.forbid,
                source = EXCLUDED.source,
                seed = EXCLUDED.seed,
                enabled = EXCLUDED.enabled,
                updated_at = EXCLUDED.updated_at,
                msg_embedding = COALESCE(intent_math.msg_embedding, EXCLUDED.msg_embedding)
        """
        with self.pool.connection() as conn:
            conn.execute(
                sql,
                {
                    "msg": row.msg[:1024],
                    "tokens": tokens,
                    "emb": _embedding_literal(row.msg_embedding),
                    "score_limit": row.score_limit,
                    "skill_id": row.skill_id,
                    "skill_key": row.skill_id or "",
                    "skill_level": row.skill_level,
                    "answer": row.answer,
                    "forbid": row.forbid,
                    "source": row.source,
                    "seed": row.seed,
                    "enabled": row.enabled,
                    "ts": utcnow(),
                },
            )
            conn.commit()

    def fill_embedding(self, row_id: int, vec: list[float]) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE intent_math SET msg_embedding = %s::vector, updated_at = %s "
                "WHERE id = %s",
                (_embedding_literal(vec), utcnow(), row_id),
            )
            conn.commit()

    def list_missing_embeddings(self, limit: int = 200) -> list[IntentMathRow]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM intent_math WHERE enabled AND msg_embedding IS NULL "
                "ORDER BY id LIMIT %s",
                (limit,),
            ).fetchall()
        return [row_to_intent(r) for r in rows]

    def prune_missing_skills(self, valid_ids: set[str]) -> None:
        with self.pool.connection() as conn:
            if valid_ids:
                conn.execute(
                    "DELETE FROM intent_math WHERE forbid = 0 "
                    "AND skill_id IS NOT NULL AND skill_id <> ALL(%s)",
                    (list(valid_ids),),
                )
            else:
                conn.execute(
                    "DELETE FROM intent_math WHERE forbid = 0 AND skill_id IS NOT NULL"
                )
            conn.commit()

    def prune_forbid_seeds(self, valid_seeds: set[str]) -> None:
        with self.pool.connection() as conn:
            if valid_seeds:
                conn.execute(
                    "DELETE FROM intent_math WHERE forbid = 1 AND seed <> ALL(%s)",
                    (list(valid_seeds),),
                )
            else:
                conn.execute("DELETE FROM intent_math WHERE forbid = 1")
            conn.commit()

    def _scope_sql(
        self, forbid: int, skill_level: int, skill_ids: set[str] | None
    ) -> tuple[str, list[Any]]:
        where = ["enabled = TRUE", "forbid = %s", "skill_level = %s"]
        params: list[Any] = [forbid, skill_level]
        if skill_ids is not None:
            where.append("skill_id = ANY(%s)")
            params.append(list(skill_ids))
        return " AND ".join(where), params

    def keyword_search(
        self,
        tsquery: str,
        *,
        forbid: int,
        skill_level: int,
        skill_ids: set[str] | None,
        limit: int,
    ) -> list[IntentMathRow]:
        if not tsquery.strip():
            return []
        where, params = self._scope_sql(forbid, skill_level, skill_ids)
        sql = (
            f"SELECT * FROM intent_math WHERE {where} "
            "AND msg_tsv @@ to_tsquery('simple', %s) "
            "ORDER BY ts_rank(msg_tsv, to_tsquery('simple', %s)) DESC LIMIT %s"
        )
        with self.pool.connection() as conn:
            rows = conn.execute(sql, [*params, tsquery, tsquery, limit]).fetchall()
        return [row_to_intent(r) for r in rows]

    def vector_search(
        self,
        embedding: list[float],
        *,
        forbid: int,
        skill_level: int,
        skill_ids: set[str] | None,
        limit: int,
    ) -> list[IntentMathRow]:
        if not embedding:
            return []
        where, params = self._scope_sql(forbid, skill_level, skill_ids)
        sql = (
            f"SELECT * FROM intent_math WHERE {where} "
            "AND msg_embedding IS NOT NULL "
            "ORDER BY msg_embedding <=> %s::vector LIMIT %s"
        )
        try:
            with self.pool.connection() as conn:
                rows = conn.execute(
                    sql, [*params, _embedding_literal(embedding), limit]
                ).fetchall()
        except Exception:
            return []
        return [row_to_intent(r) for r in rows]
