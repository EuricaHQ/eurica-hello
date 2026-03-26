from __future__ import annotations

from machine.states import State
from machine.context import DecisionContext


# In-memory store: decision_id → (state, context)
_store: dict[str, tuple[State, DecisionContext]] = {}


def load(decision_id: str) -> tuple[State, DecisionContext]:
    """Load state and context for a decision. Creates new if not found."""
    if decision_id not in _store:
        ctx = DecisionContext(
            decision_id=decision_id,
            participants=[],  # populated by first message
            responses={},
        )
        _store[decision_id] = (State.CLARIFYING, ctx)
    return _store[decision_id]


def save(decision_id: str, state: State, context: DecisionContext) -> None:
    """Persist updated state and context."""
    _store[decision_id] = (state, context)


def get_all() -> dict[str, tuple[State, DecisionContext]]:
    """Return all decisions (debug/inspection only)."""
    return dict(_store)
