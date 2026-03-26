from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    ASK_QUESTION = "ask_question"
    AGGREGATE = "aggregate"
    VALIDATE_CONSTRAINT = "validate_constraint"
    RESOLVE_CONFLICT = "resolve_conflict"
    PROPOSE_DECISION = "propose_decision"
    FINALIZE_DECISION = "finalize_decision"
    MARK_INFEASIBLE = "mark_infeasible"
    ADDRESS_AVOIDANCE = "address_avoidance"


@dataclass
class Action:
    """A typed, side-effect-free action emitted by the machine.

    Actions describe WHAT should happen, not HOW.
    Execution is handled by a separate layer (not yet implemented).
    """

    type: ActionType
    payload: dict | None = None
