from __future__ import annotations

from dataclasses import dataclass, field

from machine.events import Event


@dataclass
class Signals:
    """Structured output from LLM interpretation.

    The LLM produces Signals, NOT Events directly.
    Signal-to-Event mapping happens below, not in the LLM.

    Extensible: new signal fields are added here as the
    interpreter learns to extract more structure.
    """

    # Content signals
    preferences: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)

    # Conflict signals
    conflict_signal: bool = False
    objections: list[str] = field(default_factory=list)

    # Decision signals
    confirmation: bool = False
    rejection: bool = False
    rejection_reason: str | None = None

    # Avoidance signals
    avoidance_signal: bool = False


def map_signals_to_event(signals: Signals) -> Event:
    """Deterministic mapping from structured signals to a machine event.

    This is the boundary between interpretation and control.
    Rules are explicit, ordered, and testable — no LLM involvement.

    Priority order (first match wins):
    1. Explicit decision confirmation/rejection
    2. Avoidance detection
    3. Conflict / objection
    4. Constraint presence → validation required
    5. Default → response received
    """
    if signals.confirmation:
        return Event.DECISION_CONFIRMED

    if signals.rejection:
        return Event.DECISION_REJECTED

    if signals.avoidance_signal:
        return Event.AVOIDANCE_DETECTED

    # Conflict and constraint signals are NOT separate events (spec v2.9).
    # They are merged into context and assessed by guards on
    # AGGREGATION_COMPLETED. All content input maps to RESPONSE_RECEIVED.

    return Event.RESPONSE_RECEIVED
