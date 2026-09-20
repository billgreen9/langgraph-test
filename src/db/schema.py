"""建表 DDL 与旧表检测（仅供 Database.init_schema 使用）。"""

from __future__ import annotations

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_record (
    id          BIGSERIAL PRIMARY KEY,
    chat_id     VARCHAR(64) UNIQUE NOT NULL,
    session_id  VARCHAR(64) NOT NULL,
    user_id     VARCHAR(64),
    role        VARCHAR(16) NOT NULL DEFAULT 'user',
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chat_record_session ON chat_record(session_id, created_at);

CREATE TABLE IF NOT EXISTS chat_task (
    id              BIGSERIAL PRIMARY KEY,
    task_id         VARCHAR(64) UNIQUE NOT NULL,
    session_id      VARCHAR(64) NOT NULL,
    title           VARCHAR(255) NOT NULL DEFAULT '',
    content         TEXT NOT NULL DEFAULT '',
    entry_skill_id  VARCHAR(255) NOT NULL,
    arguments       JSONB NOT NULL DEFAULT '{}'::jsonb,
    status          VARCHAR(20) NOT NULL DEFAULT 'collecting',
    pause_requested BOOLEAN NOT NULL DEFAULT FALSE,
    output          TEXT,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chat_task_session ON chat_task(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_task_status  ON chat_task(status);

CREATE TABLE IF NOT EXISTS chat_task_message (
    id         BIGSERIAL PRIMARY KEY,
    task_id    VARCHAR(64) NOT NULL REFERENCES chat_task(task_id),
    chat_id    VARCHAR(64) NOT NULL REFERENCES chat_record(chat_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (task_id, chat_id)
);
CREATE INDEX IF NOT EXISTS idx_ctm_chat ON chat_task_message(chat_id);
CREATE INDEX IF NOT EXISTS idx_ctm_task ON chat_task_message(task_id);

CREATE TABLE IF NOT EXISTS skill_registry (
    skill_id     VARCHAR(255) PRIMARY KEY,
    name         VARCHAR(255) NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    skill_type   VARCHAR(20) NOT NULL,
    level        INTEGER NOT NULL,
    parent_id    VARCHAR(255),
    fs_path      TEXT NOT NULL,
    keywords     JSONB NOT NULL DEFAULT '[]'::jsonb,
    has_children BOOLEAN NOT NULL DEFAULT FALSE,
    prefetched   BOOLEAN NOT NULL DEFAULT FALSE,
    manifest     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_skill_registry_level ON skill_registry(level);
CREATE INDEX IF NOT EXISTS idx_skill_registry_parent ON skill_registry(parent_id);
"""

INTENT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS intent_config (
    key        VARCHAR(64) PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS intent_math (
    id             BIGSERIAL PRIMARY KEY,
    msg            VARCHAR(1024) NOT NULL,
    msg_tsv        tsvector,
    msg_embedding  vector,
    score_limit    DOUBLE PRECISION NOT NULL DEFAULT 0.45,
    skill_id       VARCHAR(255),
    skill_key      VARCHAR(255) NOT NULL DEFAULT '',
    skill_level    INTEGER NOT NULL DEFAULT 0,
    answer         TEXT NOT NULL DEFAULT '',
    forbid         SMALLINT NOT NULL DEFAULT 0,
    source         VARCHAR(32) NOT NULL DEFAULT 'seed',
    seed           VARCHAR(255) NOT NULL DEFAULT '',
    enabled        BOOLEAN NOT NULL DEFAULT TRUE,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (msg, skill_key)
);
CREATE INDEX IF NOT EXISTS intent_math_msg_gin ON intent_math USING gin(msg_tsv);
CREATE INDEX IF NOT EXISTS intent_math_skill ON intent_math(skill_id);
CREATE INDEX IF NOT EXISTS intent_math_level ON intent_math(skill_level);
CREATE INDEX IF NOT EXISTS intent_math_forbid_level
    ON intent_math(forbid, skill_level) WHERE enabled;
"""

INTENT_VECTOR_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS intent_math_embedding_hnsw
    ON intent_math USING hnsw (msg_embedding vector_cosine_ops)
    WHERE msg_embedding IS NOT NULL;
"""

# 旧版单体 chat_records 表（含 status/response/pause_requested 列）的特征列
LEGACY_CHECK_SQL = """
SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'chat_records' AND column_name = 'status'
) AS is_legacy
"""
