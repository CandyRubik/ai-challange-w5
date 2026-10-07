from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from threading import Lock
from typing import Callable, Protocol
from uuid import uuid4

from ..memory.messages import StoredMessage
from ..state.task import StoredTask, TaskContext, TaskConflict, TaskState
from ..orchestration.profiles import (
    DEFAULT_PROFILE_ID,
    ensure_profile_schema,
)


DEFAULT_CHAT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "chat.sqlite3"
DEFAULT_DB_PATH = DEFAULT_CHAT_DB_PATH


class ChatSessionNotFound(LookupError):
    pass


class ChatTurnConflict(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StoredSession:
    id: str
    profile_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: tuple[StoredMessage, ...] = ()
    task: StoredTask | None = None
    provider: str = "deepseek"


class ChatSessionRepository(Protocol):
    def create(self, profile_id: str = DEFAULT_PROFILE_ID, *, provider: str = "deepseek") -> StoredSession: ...

    def set_provider(self, session_id: str, provider: str) -> StoredSession: ...

    def list(self, profile_id: str | None = None) -> list[StoredSession]: ...

    def get(self, session_id: str) -> StoredSession: ...

    def clear(self, profile_id: str | None = None) -> None: ...

    def append_exchange(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        *, provider: str | None = None, model: str | None = None, generated: bool = True,
    ) -> StoredSession: ...

    def append_command(self, session_id: str, command_text: str) -> StoredSession: ...

    def append_refusal(
        self, session_id: str, user_content: str, assistant_content: str,
    ) -> StoredSession: ...

    def start_turn(self, session_id: str, content: str, *, provider: str | None = None, model: str | None = None) -> StoredMessage: ...

    def save_memory_update(self, session_id: str, message_id: str, update: str) -> None: ...

    def fail_turn(self, session_id: str, message_id: str) -> None: ...

    def finish_turn(
        self, session_id: str, message_id: str, answer: str, *, refusal: bool = False,
        on_commit: Callable[[sqlite3.Connection], None] | None = None,
    ) -> StoredSession: ...


    def task_operation(self, session_id: str) -> AbstractContextManager[None]: ...

    def create_task(
        self, session_id: str, task: TaskContext, *, assistant_content: str | None = None,
    ) -> StoredSession: ...

    def update_task(
        self, session_id: str, task: TaskContext,
        user_content: str, assistant_content: str, *,
        expected_revision: int | None = None,
        expected_progress_revision: int | None = None,
        pause_only: bool = False,
        provider: str | None = None, model: str | None = None,
    ) -> StoredSession: ...


class SQLiteChatSessionRepository:
    """Durable chat history isolated behind a repository boundary."""

    def __init__(self, database_path: str | Path = DEFAULT_CHAT_DB_PATH) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._task_locks: dict[str, Lock] = {}
        self._task_locks_guard = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
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
            ensure_profile_schema(connection)
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL DEFAULT 'default'
                        REFERENCES user_profiles(id),
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    kind TEXT NOT NULL DEFAULT 'message'
                        CHECK (kind IN ('message', 'command')),
                    content TEXT NOT NULL,
                    is_refusal INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    UNIQUE (session_id, position)
                );

                CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                ON chat_messages(session_id, position);

                CREATE TABLE IF NOT EXISTS chat_tasks (
                    session_id TEXT PRIMARY KEY REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    context TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    progress_revision INTEGER NOT NULL DEFAULT 0
                );
                """,
            )
            session_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(chat_sessions)")
            }
            if "profile_id" not in session_columns:
                connection.execute(
                    """
                    ALTER TABLE chat_sessions
                    ADD COLUMN profile_id TEXT REFERENCES user_profiles(id)
                    """,
                )
                connection.execute(
                    "UPDATE chat_sessions SET profile_id = ? WHERE profile_id IS NULL",
                    (DEFAULT_PROFILE_ID,),
                )
            if "provider" not in session_columns:
                connection.execute("ALTER TABLE chat_sessions ADD COLUMN provider TEXT NOT NULL DEFAULT 'deepseek'")
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_profile
                ON chat_sessions(profile_id, updated_at)
                """,
            )

            message_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(chat_messages)")
            }
            if "kind" not in message_columns:
                connection.execute(
                    """
                    ALTER TABLE chat_messages
                    ADD COLUMN kind TEXT NOT NULL DEFAULT 'message'
                        CHECK (kind IN ('message', 'command'))
                    """,
                )
            if "is_refusal" not in message_columns:
                connection.execute(
                    "ALTER TABLE chat_messages ADD COLUMN is_refusal INTEGER NOT NULL DEFAULT 0",
                )
            for column, declaration in (
                ("status", "TEXT NOT NULL DEFAULT 'done'"), ("error", "TEXT"), ("memory_update", "TEXT"),
                ("provider", "TEXT"), ("model", "TEXT"),
            ):
                if column not in message_columns:
                    connection.execute(f"ALTER TABLE chat_messages ADD COLUMN {column} {declaration}")

    @staticmethod
    def _timestamp(value: datetime) -> str:
        return value.isoformat()

    @staticmethod
    def _datetime(value: str) -> datetime:
        return datetime.fromisoformat(value)

    def _load_session(
        self,
        connection: sqlite3.Connection,
        session_id: str,
    ) -> StoredSession:
        row = connection.execute(
            """
            SELECT id, profile_id, title, created_at, updated_at, provider
            FROM chat_sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            raise ChatSessionNotFound(session_id)

        message_rows = connection.execute(
            """
            SELECT id, role, kind, content, created_at, is_refusal, status, error, memory_update, provider, model
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY position
            """,
            (session_id,),
        ).fetchall()
        messages = tuple(
            StoredMessage(
                id=message["id"],
                role=message["role"],
                kind=message["kind"],
                content=message["content"],
                created_at=self._datetime(message["created_at"]),
                refusal=bool(message["is_refusal"]),
                status=message["status"], error=message["error"], memory_update=message["memory_update"],
                provider=message["provider"], model=message["model"],
            )
            for message in message_rows
        )
        task_row = connection.execute(
            "SELECT context, revision, progress_revision FROM chat_tasks WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        task = None if task_row is None else StoredTask(
            context=TaskContext.from_json(task_row["context"]),
            revision=task_row["revision"],
            progress_revision=task_row["progress_revision"],
        )
        return StoredSession(
            id=row["id"],
            profile_id=row["profile_id"] or DEFAULT_PROFILE_ID,
            title=row["title"],
            created_at=self._datetime(row["created_at"]),
            updated_at=self._datetime(row["updated_at"]),
            messages=messages,
            task=task,
            provider=row["provider"],
        )

    def create(self, profile_id: str = DEFAULT_PROFILE_ID, *, provider: str = "deepseek") -> StoredSession:
        session_id = str(uuid4())
        now = datetime.now(timezone.utc)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO chat_sessions
                    (id, profile_id, title, created_at, updated_at, provider)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    profile_id,
                    "Новый чат",
                    self._timestamp(now),
                    self._timestamp(now),
                    provider,
                ),
            )
        return StoredSession(session_id, profile_id, "Новый чат", now, now, provider=provider)

    def set_provider(self, session_id: str, provider: str) -> StoredSession:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._load_session(connection, session_id)
            connection.execute("UPDATE chat_sessions SET provider = ? WHERE id = ?", (provider, session_id))
        return self.get(session_id)

    def list(self, profile_id: str | None = None) -> list[StoredSession]:
        with self._connection() as connection:
            connection.execute("BEGIN")
            if profile_id is None:
                rows = connection.execute(
                    """
                    SELECT id FROM chat_sessions
                    ORDER BY updated_at DESC
                    LIMIT 100
                    """,
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT id FROM chat_sessions
                    WHERE profile_id = ?
                    ORDER BY updated_at DESC
                    LIMIT 100
                    """,
                    (profile_id,),
                ).fetchall()
            return [self._load_session(connection, row["id"]) for row in rows]

    def get(self, session_id: str) -> StoredSession:
        with self._connection() as connection:
            connection.execute("BEGIN")
            return self._load_session(connection, session_id)

    def clear(self, profile_id: str | None = None) -> None:
        with self._connection() as connection:
            if profile_id is None:
                connection.execute("DELETE FROM chat_sessions")
            else:
                connection.execute(
                    "DELETE FROM chat_sessions WHERE profile_id = ?",
                    (profile_id,),
                )

    def append_exchange(
        self,
        session_id: str,
        user_content: str,
        assistant_content: str,
        *, provider: str | None = None, model: str | None = None, generated: bool = True,
    ) -> StoredSession:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = self._load_session(connection, session_id)
            self._append_exchange(connection, session, user_content, assistant_content,
                                  provider=provider, model=model, generated=generated)

        return self.get(session_id)

    def append_command(self, session_id: str, command_text: str) -> StoredSession:
        now = datetime.now(timezone.utc)
        timestamp = self._timestamp(now)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = self._load_session(connection, session_id)
            self._require_resolved(session)
            position = len(session.messages)
            title = session.title
            if not session.messages:
                title = command_text.replace("\n", " ").strip()[:60] or "Новый чат"
            connection.execute(
                """
                INSERT INTO chat_messages
                    (id, session_id, position, role, kind, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()), session_id, position, "user", "command",
                    command_text, timestamp,
                ),
            )
            connection.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, timestamp, session_id),
            )
        return self.get(session_id)

    @staticmethod
    def _require_resolved(session: StoredSession) -> None:
        if session.messages and session.messages[-1].status != "done":
            raise ChatTurnConflict("Сначала повторите последний незавершённый ответ")

    def start_turn(self, session_id: str, content: str, *, provider: str | None = None, model: str | None = None) -> StoredMessage:
        message_id = str(uuid4())
        timestamp = self._timestamp(datetime.now(timezone.utc))
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = self._load_session(connection, session_id)
            self._require_resolved(session)
            connection.execute(
                """INSERT INTO chat_messages (id, session_id, position, role, kind, content, created_at, status, provider, model)
                   VALUES (?, ?, ?, 'user', 'message', ?, ?, 'pending', ?, ?)""",
                (message_id, session_id, len(session.messages), content.strip(), timestamp, provider, model),
            )
            connection.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (session.title if session.messages else content.replace("\n", " ").strip()[:60], timestamp, session_id),
            )
        return self.get(session_id).messages[-1]

    def save_memory_update(self, session_id: str, message_id: str, update: str) -> None:
        with self._connection() as connection:
            changed = connection.execute(
                """UPDATE chat_messages SET memory_update = ?
                   WHERE id = ? AND session_id = ? AND status != 'done' AND memory_update IS NULL""",
                (update, message_id, session_id),
            ).rowcount
            if changed != 1:
                raise ChatTurnConflict("Сообщение уже обработано")

    def fail_turn(self, session_id: str, message_id: str) -> None:
        with self._connection() as connection:
            connection.execute(
                """UPDATE chat_messages SET status = 'failed', error = 'Не удалось ответить. Повторите попытку.'
                   WHERE id = ? AND session_id = ? AND status != 'done'""",
                (message_id, session_id),
            )

    def finish_turn(
        self, session_id: str, message_id: str, answer: str, *, refusal: bool = False,
        on_commit: Callable[[sqlite3.Connection], None] | None = None,
    ) -> StoredSession:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = self._load_session(connection, session_id)
            if not session.messages or session.messages[-1].id != message_id or session.messages[-1].status == "done":
                raise ChatTurnConflict("Повторить можно только последнее сообщение без ответа")
            if on_commit is not None:
                on_commit(connection)
            timestamp = self._timestamp(datetime.now(timezone.utc))
            connection.execute(
                "UPDATE chat_messages SET status = 'done', error = NULL, is_refusal = ? WHERE id = ?",
                (int(refusal), message_id),
            )
            connection.execute(
                """INSERT INTO chat_messages (id, session_id, position, role, content, created_at, is_refusal, provider, model)
                   VALUES (?, ?, ?, 'assistant', ?, ?, ?, ?, ?)""",
                (str(uuid4()), session_id, len(session.messages), answer, timestamp, int(refusal),
                 None if refusal else session.messages[-1].provider,
                 None if refusal else session.messages[-1].model),
            )
            connection.execute("UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (timestamp, session_id))
        return self.get(session_id)

    def append_refusal(
        self, session_id: str, user_content: str, assistant_content: str,
    ) -> StoredSession:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = self._load_session(connection, session_id)
            self._append_exchange(
                connection, session, user_content, assistant_content, refusal=True,
            )
        return self.get(session_id)


    def _append_exchange(
        self, connection: sqlite3.Connection, session: StoredSession,
        user_content: str, assistant_content: str,
        *, refusal: bool = False, provider: str | None = None, model: str | None = None, generated: bool = True,
    ) -> None:
        timestamp = self._timestamp(datetime.now(timezone.utc))
        position = len(session.messages)
        title = session.title
        if not session.messages:
            title = user_content.replace("\n", " ").strip()[:60] or "Новый чат"
        connection.executemany(
            """
            INSERT INTO chat_messages
                (id, session_id, position, role, content, created_at, is_refusal, provider, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (str(uuid4()), session.id, position, "user", user_content, timestamp, int(refusal), provider, model),
                (str(uuid4()), session.id, position + 1, "assistant", assistant_content, timestamp, int(refusal),
                 provider if generated else None, model if generated else None),
            ],
        )
        connection.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, timestamp, session.id),
        )

    @contextmanager
    def task_operation(self, session_id: str) -> Iterator[None]:
        # The repository is shared by requests. Pause/resume deliberately bypass
        # this lock; a running generation can finish into a paused snapshot.
        with self._task_locks_guard:
            lock = self._task_locks.setdefault(session_id, Lock())
        if not lock.acquire(blocking=False):
            raise TaskConflict("Шаг уже выполняется. Дождитесь ответа агента")
        try:
            yield
        finally:
            lock.release()

    def create_task(
        self, session_id: str, task: TaskContext, *, assistant_content: str | None = None,
    ) -> StoredSession:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = self._load_session(connection, session_id)
            self._require_resolved(session)
            if session.task is not None:
                raise TaskConflict("В этом чате уже есть задача. Создайте новый чат")
            connection.execute(
                "INSERT INTO chat_tasks (session_id, context) VALUES (?, ?)",
                (session_id, task.to_json()),
            )
            self._append_exchange(
                connection, session, "Задача: " + task.task,
                assistant_content if assistant_content is not None else
                "Задача создана. Нажмите «Сформировать план», затем утвердите его.",
            )
        return self.get(session_id)

    def update_task(
        self, session_id: str, task: TaskContext,
        user_content: str, assistant_content: str, *,
        expected_revision: int | None = None,
        expected_progress_revision: int | None = None,
        pause_only: bool = False,
        provider: str | None = None, model: str | None = None,
    ) -> StoredSession:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = self._load_session(connection, session_id)
            stored = session.task
            if stored is None:
                raise TaskConflict("В чате нет задачи")
            if expected_revision is not None and stored.revision != expected_revision:
                raise TaskConflict("Состояние изменилось. Обновите чат и повторите действие")
            if expected_progress_revision is not None:
                if stored.progress_revision != expected_progress_revision:
                    raise TaskConflict("Состояние задачи изменилось во время выполнения")
                # A pause arriving during successful validation is acknowledged
                # as completed: DONE is terminal and cannot remain paused.
                task = replace(
                    task, paused=stored.context.paused and task.state != TaskState.DONE,
                )
            connection.execute(
                """
                UPDATE chat_tasks
                SET context = ?, revision = revision + 1,
                    progress_revision = progress_revision + ?
                WHERE session_id = ?
                """,
                (task.to_json(), 0 if pause_only else 1, session_id),
            )
            self._append_exchange(connection, session, user_content, assistant_content, provider=provider, model=model)
        return self.get(session_id)
