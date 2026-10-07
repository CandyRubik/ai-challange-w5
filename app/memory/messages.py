from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class StoredMessage:
    id: str
    role: str
    kind: str
    content: str
    created_at: datetime
    refusal: bool = False
    status: str = "done"
    error: str | None = None
    memory_update: str | None = None
