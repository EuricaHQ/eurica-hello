from __future__ import annotations

from machine.events import Event


def map_signals_to_event(signals: dict) -> Event:
    """Deterministic mapping from LLM signals (dict) to a machine event.

    This is the boundary between interpretation and control.
    Rules are explicit, ordered, and testable — no LLM involvement.

    Only two outcomes:
    - AVOIDANCE_DETECTED (if avoidance signal present)
    - RESPONSE_RECEIVED (everything else)

    All other event types (DECISION_CONFIRMED, DECISION_REJECTED, etc.)
    are triggered by the system or explicit user actions, NOT by
    signal interpretation.
    """
    if signals.get("avoidance", False):
        return Event.AVOIDANCE_DETECTED

    return Event.RESPONSE_RECEIVED
