"""Tests for LLM advisory layer on has_critical_unresolved_participants.

LLM is injected via ctx.services["llm"] (dependency injection).
No global state. No real API calls.

Verifies:
- rule-based False + LLM says critical → True (escalation)
- rule-based True → LLM not called (short-circuit)
- LLM failure → fallback to rule-based result
- LLM returns invalid data → safe fallback
- no LLM in services → rule-based only

Run: python3 test_critical_llm_advisory.py
"""

from machine.context import DecisionContext
from machine.transition import (
    _has_critical_unresolved_participants,
    _has_critical_unresolved_rule_based,
)
from llm.interface import LLM


# ---------------------------------------------------------------------------
# Mock LLM implementations for testing
# ---------------------------------------------------------------------------

class _MockLLMEscalates(LLM):
    """Always flags the first missing participant as critical."""

    def interpret(self, message, context):
        return {}

    def generate(self, state, context):
        return ""

    def evaluate_critical_participants(self, context, missing):
        return {"critical_participants": missing[:1]}


class _MockLLMAgrees(LLM):
    """Always agrees no one is critical."""

    def interpret(self, message, context):
        return {}

    def generate(self, state, context):
        return ""

    def evaluate_critical_participants(self, context, missing):
        return {"critical_participants": []}


class _MockLLMCrashes(LLM):
    """Always raises an exception."""

    def interpret(self, message, context):
        return {}

    def generate(self, state, context):
        return ""

    def evaluate_critical_participants(self, context, missing):
        raise RuntimeError("API timeout")


class _MockLLMBadType(LLM):
    """Returns non-list for critical_participants."""

    def interpret(self, message, context):
        return {}

    def generate(self, state, context):
        return ""

    def evaluate_critical_participants(self, context, missing):
        return {"critical_participants": "eve"}


class _MockLLMNoKey(LLM):
    """Returns dict without expected key."""

    def interpret(self, message, context):
        return {}

    def generate(self, state, context):
        return ""

    def evaluate_critical_participants(self, context, missing):
        return {"other_key": ["eve"]}


class _MockLLMReturnsNone(LLM):
    """Returns None instead of dict."""

    def interpret(self, message, context):
        return {}

    def generate(self, state, context):
        return ""

    def evaluate_critical_participants(self, context, missing):
        return None


class _MockLLMTracked(LLM):
    """Tracks whether evaluate_critical_participants was called."""

    def __init__(self):
        self.call_count = 0

    def interpret(self, message, context):
        return {}

    def generate(self, state, context):
        return ""

    def evaluate_critical_participants(self, context, missing):
        self.call_count += 1
        return {"critical_participants": []}


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def _majority_locked_ctx(**overrides):
    """Rule-based returns False (majority locked: Italian 3/5)."""
    base = dict(
        question="Where to eat?",
        participants=["alice", "bob", "carol", "dave", "eve"],
        min_participants=3,
        decision_rule="majority",
        responses={
            "alice": ["Italian"], "bob": ["Italian"],
            "carol": ["Italian"], "dave": ["Thai"],
        },
        preferences=["Italian", "Italian", "Italian", "Thai"],
        services={},
    )
    base.update(overrides)
    return DecisionContext(**base)


def _consent_missing_ctx(**overrides):
    """Rule-based returns True (consent + missing participant)."""
    base = dict(
        question="Where to eat?",
        participants=["alice", "bob", "carol"],
        min_participants=2,
        decision_rule="consent",
        responses={"alice": ["Italian"], "carol": ["Thai"]},
        preferences=["Italian", "Thai"],
        services={},
    )
    base.update(overrides)
    return DecisionContext(**base)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_llm_escalates_when_rule_says_false():
    """LLM flags eve as critical despite majority being locked."""
    ctx = _majority_locked_ctx(services={"llm": _MockLLMEscalates()})

    assert not _has_critical_unresolved_rule_based(ctx)
    assert _has_critical_unresolved_participants(ctx) is True


def test_llm_agrees_not_critical():
    """LLM agrees no one is critical → result stays False."""
    ctx = _majority_locked_ctx(services={"llm": _MockLLMAgrees()})

    assert _has_critical_unresolved_participants(ctx) is False


def test_llm_not_called_when_rule_says_true():
    """When rule-based returns True, LLM should never be invoked."""
    mock = _MockLLMTracked()
    ctx = _consent_missing_ctx(services={"llm": mock})

    assert _has_critical_unresolved_rule_based(ctx) is True
    result = _has_critical_unresolved_participants(ctx)
    assert result is True
    assert mock.call_count == 0, f"LLM was called {mock.call_count} times, expected 0"


def test_llm_failure_falls_back():
    """LLM crashes → guard returns rule-based result (False)."""
    ctx = _majority_locked_ctx(services={"llm": _MockLLMCrashes()})

    assert _has_critical_unresolved_participants(ctx) is False


def test_llm_returns_invalid_type():
    """LLM returns non-list → treated as empty → False."""
    ctx = _majority_locked_ctx(services={"llm": _MockLLMBadType()})

    assert _has_critical_unresolved_participants(ctx) is False


def test_llm_returns_no_key():
    """LLM returns dict without expected key → empty → False."""
    ctx = _majority_locked_ctx(services={"llm": _MockLLMNoKey()})

    assert _has_critical_unresolved_participants(ctx) is False


def test_llm_returns_none():
    """LLM returns None → exception caught → False."""
    ctx = _majority_locked_ctx(services={"llm": _MockLLMReturnsNone()})

    assert _has_critical_unresolved_participants(ctx) is False


def test_no_llm_in_services():
    """When no LLM in services, guard uses rule-based only."""
    ctx = _majority_locked_ctx(services={})

    assert _has_critical_unresolved_participants(ctx) is False


def test_llm_returns_mixed_names():
    """LLM returns both valid and invalid names → non-empty → True."""
    ctx = _majority_locked_ctx(services={"llm": _MockLLMEscalates()})

    assert _has_critical_unresolved_participants(ctx) is True


if __name__ == "__main__":
    tests = [
        test_llm_escalates_when_rule_says_false,
        test_llm_agrees_not_critical,
        test_llm_not_called_when_rule_says_true,
        test_llm_failure_falls_back,
        test_llm_returns_invalid_type,
        test_llm_returns_no_key,
        test_llm_returns_none,
        test_no_llm_in_services,
        test_llm_returns_mixed_names,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} LLM advisory tests passed.")
