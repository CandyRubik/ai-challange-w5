"""SQLite history and task-state snapshots for the dedicated RAG chat."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Iterator
from uuid import uuid4

from ..storage.chat_sessions import DEFAULT_CHAT_DB_PATH
from .models import RagSession, RagSessionSummary, RagTaskState, RagTurn
from .state import TurnDecision


class RagChatNotFound(LookupError):
    pass


class RagTurnConflict(ValueError):
    pass


class SQLiteRagChatRepository:
    def __init__(self, database_path: str | Path = DEFAULT_CHAT_DB_PATH) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS rag_chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rag_chat_state (
                    session_id TEXT PRIMARY KEY REFERENCES rag_chat_sessions(id) ON DELETE CASCADE,
                    state_json TEXT NOT NULL,
                    revision INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rag_chat_turns (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES rag_chat_sessions(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    answer TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL CHECK (status IN ('pending', 'done', 'failed')),
                    decision_json TEXT,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    grounding_status TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(session_id, position)
                );
                CREATE INDEX IF NOT EXISTS idx_rag_chat_turns_session
                    ON rag_chat_turns(session_id, position);
            """)
            # Preserve the original cloud provenance when upgrading a W4 database.
            for table, name, declaration in (
                ("rag_chat_sessions", "provider", "TEXT NOT NULL DEFAULT 'deepseek'"),
                ("rag_chat_turns", "provider", "TEXT NOT NULL DEFAULT 'deepseek'"),
                ("rag_chat_turns", "model", "TEXT"),
                ("rag_chat_turns", "metrics_json", "TEXT NOT NULL DEFAULT '{}'"),
            ):
                if name not in {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _summary(row: sqlite3.Row) -> RagSessionSummary:
        return RagSessionSummary(
            id=row["id"], title=row["title"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            provider=row["provider"],
        )

    @staticmethod
    def _turn(row: sqlite3.Row) -> RagTurn:
        decision = (
            TurnDecision.model_validate_json(row["decision_json"])
            if row["decision_json"] else None
        )
        return RagTurn(
            id=row["id"], session_id=row["session_id"], position=row["position"],
            content=row["content"], answer=row["answer"], status=row["status"],
            kind=decision.kind if decision else None,
            question_scope=decision.question_scope if decision else None,
            search_question=decision.search_question if decision else "",
            sources=json.loads(row["sources_json"]),
            citations=json.loads(row["citations_json"]),
            grounding_status=row["grounding_status"], error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            provider=row["provider"], model=row["model"], metrics=json.loads(row["metrics_json"]),
        )

    def create(self, provider: str = "ollama") -> RagSession:
        session_id = str(uuid4())
        now = self._now()
        state = RagTaskState()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO rag_chat_sessions (id, title, created_at, updated_at, provider) VALUES (?, ?, ?, ?, ?)",
                (session_id, "Новый RAG-чат", now, now, provider),
            )
            connection.execute(
                "INSERT INTO rag_chat_state VALUES (?, ?, ?)",
                (session_id, state.model_dump_json(), state.revision),
            )
        return self.get(session_id)

    def list(self) -> list[RagSessionSummary]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM rag_chat_sessions ORDER BY updated_at DESC, id",
            ).fetchall()
        return [self._summary(row) for row in rows]

    def get(self, session_id: str) -> RagSession:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM rag_chat_sessions WHERE id = ?", (session_id,),
            ).fetchone()
            if row is None:
                raise RagChatNotFound(session_id)
            state_row = connection.execute(
                "SELECT state_json FROM rag_chat_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            turns = connection.execute(
                "SELECT * FROM rag_chat_turns WHERE session_id = ? ORDER BY position",
                (session_id,),
            ).fetchall()
        return RagSession(
            **self._summary(row).model_dump(),
            state=RagTaskState.model_validate_json(state_row["state_json"]),
            turns=[self._turn(turn) for turn in turns],
        )

    def delete(self, session_id: str) -> None:
        with self._connection() as connection:
            deleted = connection.execute(
                "DELETE FROM rag_chat_sessions WHERE id = ?", (session_id,),
            ).rowcount
        if not deleted:
            raise RagChatNotFound(session_id)

    def set_provider(self, session_id: str, provider: str) -> None:
        with self._connection() as connection:
            if not connection.execute(
                "UPDATE rag_chat_sessions SET provider = ?, updated_at = ? WHERE id = ?",
                (provider, self._now(), session_id),
            ).rowcount:
                raise RagChatNotFound(session_id)

    def start_turn(self, session_id: str, content: str, *, provider: str = "ollama", model: str | None = None) -> RagTurn:
        now = self._now()
        turn_id = str(uuid4())
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = connection.execute(
                "SELECT title FROM rag_chat_sessions WHERE id = ?", (session_id,),
            ).fetchone()
            if session is None:
                raise RagChatNotFound(session_id)
            unfinished = connection.execute(
                "SELECT 1 FROM rag_chat_turns WHERE session_id = ? AND status != 'done'",
                (session_id,),
            ).fetchone()
            if unfinished:
                raise RagTurnConflict("Повторите последний ответ перед новым сообщением")
            position = connection.execute(
                "SELECT COALESCE(MAX(position), 0) + 1 FROM rag_chat_turns WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            connection.execute(
                """INSERT INTO rag_chat_turns
                   (id, session_id, position, content, status, created_at, updated_at, provider, model)
                   VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)""",
                (turn_id, session_id, position, content.strip(), now, now, provider, model),
            )
            title = content.strip()[:72] if position == 1 else session["title"]
            connection.execute(
                "UPDATE rag_chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, session_id),
            )
            row = connection.execute(
                "SELECT * FROM rag_chat_turns WHERE id = ?", (turn_id,),
            ).fetchone()
        return self._turn(row)

    def save_decision(
        self, turn_id: str, decision: TurnDecision, state: RagTaskState,
        expected_revision: int,
    ) -> RagTurn:
        now = self._now()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM rag_chat_turns WHERE id = ?", (turn_id,),
            ).fetchone()
            if row is None:
                raise RagChatNotFound(turn_id)
            if row["decision_json"] is not None:
                raise RagTurnConflict("Уточнение уже применено")
            changed = connection.execute(
                """UPDATE rag_chat_state SET state_json = ?, revision = ?
                   WHERE session_id = ? AND revision = ?""",
                (state.model_dump_json(), state.revision, row["session_id"], expected_revision),
            ).rowcount
            if not changed:
                raise RagTurnConflict("Память задачи изменилась параллельно")
            connection.execute(
                "UPDATE rag_chat_turns SET decision_json = ?, updated_at = ? WHERE id = ?",
                (decision.model_dump_json(), now, turn_id),
            )
            updated = connection.execute(
                "SELECT * FROM rag_chat_turns WHERE id = ?", (turn_id,),
            ).fetchone()
        return self._turn(updated)

    def decision(self, turn_id: str) -> TurnDecision | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT decision_json FROM rag_chat_turns WHERE id = ?", (turn_id,),
            ).fetchone()
        if row is None:
            raise RagChatNotFound(turn_id)
        return TurnDecision.model_validate_json(row["decision_json"]) if row["decision_json"] else None

    def finish_turn(
        self, turn_id: str, answer: str, *, sources: list[dict], citations: list[dict],
        grounding_status: str, metrics: dict | None = None,
    ) -> RagTurn:
        now = self._now()
        with self._connection() as connection:
            connection.execute(
                """UPDATE rag_chat_turns SET answer = ?, status = 'done',
                   sources_json = ?, citations_json = ?, grounding_status = ?,
                   error = NULL, updated_at = ?, metrics_json = ? WHERE id = ?""",
                (answer, json.dumps(sources, ensure_ascii=False),
                 json.dumps(citations, ensure_ascii=False), grounding_status, now,
                 json.dumps(metrics or {}), turn_id),
            )
            row = connection.execute(
                "SELECT * FROM rag_chat_turns WHERE id = ?", (turn_id,),
            ).fetchone()
            connection.execute(
                "UPDATE rag_chat_sessions SET updated_at = ? WHERE id = ?",
                (now, row["session_id"]),
            )
        return self._turn(row)

    def fail_turn(self, turn_id: str, error: str, *, metrics: dict | None = None) -> RagTurn:
        now = self._now()
        with self._connection() as connection:
            connection.execute(
                """UPDATE rag_chat_turns SET status = 'failed', error = ?,
                   updated_at = ?, metrics_json = ? WHERE id = ?""",
                (error, now, json.dumps(metrics or {}), turn_id),
            )
            row = connection.execute(
                "SELECT * FROM rag_chat_turns WHERE id = ?", (turn_id,),
            ).fetchone()
        return self._turn(row)
