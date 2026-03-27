"""Tests for has_critical_unresolved_participants guard (spec v2.9.2).

Verifies:
- decision blocked when critical participants haven't responded
- decision allowed when missing participant is NOT critical (stable decision)
- decision allowed when all participants have responded

Run: python3 test_critical_unresolved.py
"""

from dataclasses import replace
from machine.states import State
from machine.events import Event
from machine.context import DecisionContext
from machine.transition import (
    transition,
    _has_critical_unresolved_participants,
    _solution_found,
)


def _step(state, event, ctx, expected_state, label=""):
    next_state, actions, ctx = transition(state, event, ctx)
    tag = f" ({label})" if label else ""
    assert next_state == expected_state, (
        f"{tag} expected {expected_state.value}, got {next_state.value}"
    )
    return next_state, actions, ctx


# ---------------------------------------------------------------------------
# Guard unit tests
# ---------------------------------------------------------------------------

def test_guard_all_responded():
    """No critical unresolved when everyone has responded."""
    ctx = DecisionContext(
        participants=["alice", "bob"],
        min_participants=2,
        responses={"alice": ["Italian"], "bob": ["Thai"]},
        preferences=["Italian", "Thai"],
    )
    assert not _has_critical_unresolved_participants(ctx)


def test_guard_missing_unstable():
    """Critical unresolved: bob hasn't responded, preferences diverge."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=2,
        responses={"alice": ["Italian"], "carol": ["Thai"]},
        preferences=["Italian", "Thai"],
    )
    # bob missing + preferences split → critical
    assert _has_critical_unresolved_participants(ctx)


def test_guard_missing_stable():
    """NOT critical: bob hasn't responded, but decision is unanimous."""
    ctx = DecisionContext(
        participants=["alice", "bob", "carol"],
        min_participants=2,
        responses={"alice": ["Italian"], "carol": ["Italian"]},
        preferences=["Italian", "Italian"],
    )
    # bob missing BUT all current preferences agree → stable → not critical
    assert not _has_critical_unresolved_participants(ctx)


def test_guard_no_preferences_yet():
    """Critical: someone responded but no preferences extracted yet."""
    ctx = DecisionContext(
        participants=["alice", "bob"],
        min_participants=1,
        responses={"alice": ["hmm, not sure"]},
        preferences=[],  # no preferences → unstable
    )
    # bob hasn't responded + no preferences → critical
    assert _has_critical_unresolved_participants(ctx)


def test_guard_no_participants_defined():
    """Edge: no participants → no one is unresolved."""
    ctx = DecisionContext(
        participants=[],
        min_participants=0,
        responses={},
        preferences=[],
    )
    assert not _has_critical_unresolved_participants(ctx)


# ---------------------------------------------------------------------------
# Integration: solution_found includes the guard
# ---------------------------------------------------------------------------

def test_solution_found_blocked_by_critical():
    """solution_found returns False when critical unresolved participants exist."""
    ctx = DecisionContext(
        question="Where to eat?",
        participants=["alice", "bob", "carol"],
        min_participants=2,
        responses={"alice": ["Italian"], "carol": ["Thai"]},
        preferences=["Italian", "Thai"],  # divergent → unstable
    )
    # participation satisfied (2 >= 2), no conflicts, no constraints,
    # but bob missing + unstable → solution NOT found
    assert not _solution_found(ctx)


def test_solution_found_allowed_when_stable():
    """solution_found returns True when missing participant is non-critical."""
    ctx = DecisionContext(
        question="Where to eat?",
        participants=["alice", "bob", "carol"],
        min_participants=2,
        responses={"alice": ["Italian"], "carol": ["Italian"]},
        preferences=["Italian", "Italian"],  # unanimous → stable
    )
    # participation satisfied (2 >= 2), unanimous, bob non-critical
    assert _solution_found(ctx)


# ---------------------------------------------------------------------------
# Transition-level: blocked decision stays in AGGREGATING fallback
# ---------------------------------------------------------------------------

def test_aggregating_blocked_by_critical_participant():
    """AGGREGATION_COMPLETED in AGGREGATING → COLLECTING (not DECIDING)
    when critical unresolved participants exist.

    This is the key behavioral test: the system does NOT jump to DECIDING
    when a missing participant could still change the outcome.
    """
    ctx = DecisionContext(
        question="Where to eat?",
        participants=["alice", "bob", "carol"],
        min_participants=2,
        responses={"alice": ["Italian"], "carol": ["Thai"]},
        preferences=["Italian", "Thai"],  # divergent
    )

    # solution_found fails (critical unresolved) → fallback to COLLECTING
    state, actions, ctx = _step(
        State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        State.COLLECTING, "blocked: critical unresolved bob",
    )


def test_aggregating_allowed_when_stable():
    """AGGREGATION_COMPLETED in AGGREGATING → DECIDING
    when missing participant is non-critical (unanimous decision).
    """
    ctx = DecisionContext(
        question="Where to eat?",
        participants=["alice", "bob", "carol"],
        min_participants=2,
        responses={"alice": ["Italian"], "carol": ["Italian"]},
        preferences=["Italian", "Italian"],  # unanimous
    )

    # solution_found passes (bob non-critical) → DECIDING
    state, actions, ctx = _step(
        State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        State.DECIDING, "allowed: bob non-critical",
    )


def test_aggregating_all_responded():
    """AGGREGATION_COMPLETED → DECIDING when everyone has responded."""
    ctx = DecisionContext(
        question="Where to eat?",
        participants=["alice", "bob"],
        min_participants=2,
        responses={"alice": ["Italian"], "bob": ["Thai"]},
        preferences=["Italian", "Thai"],  # divergent but everyone responded
    )

    # All responded → no critical unresolved → DECIDING
    state, actions, ctx = _step(
        State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        State.DECIDING, "all responded, divergent ok",
    )


if __name__ == "__main__":
    tests = [
        # Guard unit tests
        test_guard_all_responded,
        test_guard_missing_unstable,
        test_guard_missing_stable,
        test_guard_no_preferences_yet,
        test_guard_no_participants_defined,
        # solution_found integration
        test_solution_found_blocked_by_critical,
        test_solution_found_allowed_when_stable,
        # Transition-level
        test_aggregating_blocked_by_critical_participant,
        test_aggregating_allowed_when_stable,
        test_aggregating_all_responded,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} critical_unresolved tests passed.")
