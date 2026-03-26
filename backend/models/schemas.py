from __future__ import annotations

from pydantic import BaseModel


class MessageRequest(BaseModel):
    decision_id: str
    participant: str
    message: str


class MessageResponse(BaseModel):
    decision_id: str
    state: str
    reply: str
    actions_executed: list[str]
