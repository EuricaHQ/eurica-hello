"""Test: AGGREGATION_COMPLETED is always triggered in COLLECTING,
but the transition outcome depends on participation_satisfied.

Spec v2.9 section 28 — COLLECTING:
  AGGREGATION_COMPLETED:
    if participation_satisfied → AGGREGATING
    else → COLLECTING

Spec v2.9 system principle:
  Aggregation is NOT autonomous.
  The system always fires AGGREGATION_COMPLETED in COLLECTING,
  but guards control whether the state actually advances.

Setup:
  - min_participants = 3
  - participants added one at a time

Fail if:
  - system transitions to AGGREGATING before 3 participants
  - AGGREGATION_COMPLETED is not triggered in COLLECTING

Run: python3 test_aggregation_guard.py
"""

from dataclasses import replace

from machine.states import State
from machine.events import Event
from machine.context import DecisionContext
from machine.transition import transition
from api.routes import _derive_system_event


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

class EventLog:
    """Records event sequence with before/after state for inspection."""

    def __init__(self):
        self.entries: list[dict] = []

    def record(self, event: Event, state_before: State, state_after: State,
               label: str):
        self.entries.append({
            "label": label,
            "event": event.value,
            "state_before": state_before.value,
            "state_after": state_after.value,
            "changed": state_before != state_after,
        })

    def dump(self) -> str:
        lines = []
        for i, e in enumerate(self.entries):
            arrow = "→" if e["changed"] else "·"
            lines.append(
                f"  [{i+1}] {e['label']:.<50s} "
                f"{e['event']}: {e['state_before']} {arrow} {e['state_after']}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_participant(ctx: DecisionContext, name: str) -> DecisionContext:
    """Simulate a participant submitting a response."""
    responses = dict(ctx.responses)
    responses[name] = [f"{name}'s preference"]
    participants = list(ctx.participants)
    if name not in participants:
        participants.append(name)
    return replace(ctx, responses=responses, participants=participants)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_aggregation_guarded_by_participation():
    """AGGREGATION_COMPLETED always fires in COLLECTING, but only
    transitions to AGGREGATING when participation_satisfied (3/3)."""

    log = EventLog()

    ctx = DecisionContext(
        decision_id="agg-guard",
        question="Where should we eat?",
        min_participants=3,
        participants=[],
        responses={},
    )

    state = State.COLLECTING

    # --- Phase 1: only 1 participant (1/3) ---

    ctx = _add_participant(ctx, "alice")
    assert len(ctx.responses) == 1

    # RESPONSE_RECEIVED: stays in COLLECTING (spec v2.9)
    next_state, actions, ctx = transition(state, Event.RESPONSE_RECEIVED, ctx)
    log.record(Event.RESPONSE_RECEIVED, state, next_state,
               "alice responds (1/3)")
    assert next_state == State.COLLECTING, (
        f"RESPONSE_RECEIVED must stay in COLLECTING, got {next_state.value}"
    )
    state = next_state

    # System MUST derive AGGREGATION_COMPLETED in COLLECTING
    system_event = _derive_system_event(state, ctx)
    assert system_event == Event.AGGREGATION_COMPLETED, (
        f"System must trigger AGGREGATION_COMPLETED in COLLECTING, "
        f"got {system_event}"
    )

    # AGGREGATION_COMPLETED: guard blocks → stays COLLECTING
    next_state, actions, ctx = transition(state, system_event, ctx)
    log.record(system_event, state, next_state,
               "AGGREGATION_COMPLETED (1/3, blocked)")
    assert next_state == State.COLLECTING, (
        f"Must stay in COLLECTING with 1/3 participants, "
        f"got {next_state.value}"
    )
    state = next_state

    # --- Phase 2: second participant (2/3) — still not enough ---

    ctx = _add_participant(ctx, "bob")
    assert len(ctx.responses) == 2

    next_state, actions, ctx = transition(state, Event.RESPONSE_RECEIVED, ctx)
    log.record(Event.RESPONSE_RECEIVED, state, next_state,
               "bob responds (2/3)")
    assert next_state == State.COLLECTING
    state = next_state

    system_event = _derive_system_event(state, ctx)
    assert system_event == Event.AGGREGATION_COMPLETED

    next_state, actions, ctx = transition(state, system_event, ctx)
    log.record(system_event, state, next_state,
               "AGGREGATION_COMPLETED (2/3, blocked)")
    assert next_state == State.COLLECTING, (
        f"Must stay in COLLECTING with 2/3 participants, "
        f"got {next_state.value}"
    )
    state = next_state

    # --- Phase 3: third participant (3/3) — NOW it advances ---

    ctx = _add_participant(ctx, "carol")
    assert len(ctx.responses) == 3

    next_state, actions, ctx = transition(state, Event.RESPONSE_RECEIVED, ctx)
    log.record(Event.RESPONSE_RECEIVED, state, next_state,
               "carol responds (3/3)")
    assert next_state == State.COLLECTING, (
        "RESPONSE_RECEIVED must NEVER go to AGGREGATING directly"
    )
    state = next_state

    system_event = _derive_system_event(state, ctx)
    assert system_event == Event.AGGREGATION_COMPLETED, (
        f"System must trigger AGGREGATION_COMPLETED in COLLECTING, "
        f"got {system_event}"
    )

    next_state, actions, ctx = transition(state, system_event, ctx)
    log.record(system_event, state, next_state,
               "AGGREGATION_COMPLETED (3/3, passes)")
    assert next_state == State.AGGREGATING, (
        f"Must transition to AGGREGATING with 3/3 participants, "
        f"got {next_state.value}"
    )

    # --- Dump log ---
    print(f"\n  Event sequence ({len(log.entries)} steps):\n{log.dump()}")


if __name__ == "__main__":
    test_aggregation_guarded_by_participation()
    print("\n  PASS  test_aggregation_guarded_by_participation")
    print("\n1/1 tests passed.")
