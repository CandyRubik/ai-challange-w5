"""Resource limits for the single-owner, single-worker deployment."""

from dataclasses import dataclass
from collections import deque
import asyncio
import json
import math
import os
import re
from time import monotonic

from starlette.responses import JSONResponse


GENERATION_PATH = re.compile(
    r"^/api/(?:chat|rag-chat)/sessions/[^/]+/"
    r"(?:messages(?:/[^/]+/retry)?|turns(?:/[^/]+/retry)?|task/actions)$"
)


@dataclass(frozen=True)
class ServiceLimits:
    enabled: bool = False
    max_context: int = 8192
    max_output: int = 1000
    generations_per_minute: int = 12
    writes_per_minute: int = 60
    max_queue: int = 3
    queue_wait_seconds: int = 180
    max_body_bytes: int = 65536

    @classmethod
    def from_env(cls):
        limits = cls(
            enabled=os.getenv("PRIVATE_SERVICE", "0") == "1",
            **{name: int(os.getenv("PRIVATE_" + name.upper(), str(default)))
               for name, default in (
                   ("max_context", 8192), ("max_output", 1000),
                   ("generations_per_minute", 12), ("writes_per_minute", 60),
                   ("max_queue", 3), ("queue_wait_seconds", 180), ("max_body_bytes", 65536),
               )},
        )
        if (limits.max_context < 4096 or limits.max_output < 1
                or limits.max_output + 512 >= limits.max_context
                or limits.generations_per_minute < 1 or limits.writes_per_minute < 1
                or limits.max_queue < 0 or limits.queue_wait_seconds < 1 or limits.max_body_bytes < 1):
            raise ValueError("Некорректные PRIVATE_* ограничения сервиса")
        return limits


class PrivateServiceMiddleware:
    """Bound API writes and inference admission before any state or model work."""

    def __init__(self, app, *, limits: ServiceLimits | None = None):
        self.app = app
        self.limits = limits or ServiceLimits.from_env()
        self._writes: deque[float] = deque()
        self._generations: deque[float] = deque()
        self._generation_slot = asyncio.Semaphore(1)
        self._admitted = 0
        self._running_work: set[asyncio.Task] = set()

    def _rate_limit(self, history, maximum):
        now = monotonic()
        while history and history[0] <= now - 60:
            history.popleft()
        if len(history) >= maximum:
            return max(1, math.ceil(60 - (now - history[0])))
        history.append(now)
        return 0

    @staticmethod
    def _generates(scope, body):
        if scope["method"] != "POST" or not GENERATION_PATH.fullmatch(scope["path"]):
            return False
        if scope["path"].endswith("/task/actions"):
            try:
                action = json.loads(body).get("action")
            except (ValueError, AttributeError):
                return False
            return isinstance(action, str) and action not in {"pause", "resume", "approve", "replan"}
        return True

    async def __call__(self, scope, receive, send):
        if not self.limits.enabled or scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["method"] not in {"POST", "PUT", "PATCH", "DELETE"}:
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.limits.max_body_bytes:
                response = JSONResponse({"detail": "Тело запроса превышает лимит сервиса"}, 413)
                return await response(scope, receive, send)
            if not message.get("more_body", False):
                break
        retry = self._rate_limit(self._writes, self.limits.writes_per_minute)
        if not retry and self._generates(scope, body):
            retry = self._rate_limit(self._generations, self.limits.generations_per_minute)
        if retry:
            response = JSONResponse({"detail": "Слишком много запросов. Повторите позже"},
                                    429, headers={"Retry-After": str(retry)})
            return await response(scope, receive, send)
        replayed = False
        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()
        if not self._generates(scope, body):
            return await self.app(scope, replay, send)
        if self._admitted >= 1 + self.limits.max_queue:
            return await self._busy(scope, receive, send)
        self._admitted += 1
        acquired = False
        work = None
        try:
            try:
                await asyncio.wait_for(self._generation_slot.acquire(), self.limits.queue_wait_seconds)
                acquired = True
            except TimeoutError:
                return await self._busy(scope, receive, send)
            # A disconnected HTTP client must not free the slot while a sync
            # endpoint still runs its model call in Starlette's thread pool.
            work = asyncio.create_task(self.app(scope, replay, send))
            self._running_work.add(work)
            work.add_done_callback(self._work_done)
            await asyncio.shield(work)
        finally:
            # Once dispatched, the work owns admission even if the caller is
            # cancelled repeatedly. Only its completion releases the slot.
            if work is None:
                if acquired:
                    self._generation_slot.release()
                self._admitted -= 1

    def _work_done(self, work):
        self._running_work.discard(work)
        self._generation_slot.release()
        self._admitted -= 1
        if not work.cancelled():
            work.exception()  # Observe failures even after the HTTP client left.

    @staticmethod
    async def _busy(scope, receive, send):
        response = JSONResponse({"detail": "Сервис занят. Очередь заполнена или истекло время ожидания"},
                                503, headers={"Retry-After": "5"})
        await response(scope, receive, send)
