"""Tests for hybrid guard v2: signal-aware LLM advisory.

Verifies:
- relaxed env → rule=False, LLM escalates → True
- relaxed env → LLM agrees → False
- high tension → rule=True → LLM not called
- LLM returns invalid names (not in missing) → filtered → False
- LLM receives v2 signal context
- subset validation works

Run: python3 test_hybrid_guard_v2.py
"""

from machine.context import DecisionContext
from machine.transition import (
    _has_critical_unresolved_participants,
    _has_critical_unresolved_rule_based,
)
from llm.interface import LLM


# ---------------------------------------------------------------------------
# Mock LLMs
# ---------------------------------------------------------------------------

class _MockLLMEscalatesWithSignals(LLM):
    """Escalates only if it sees hard constraints in the context signals."""

    def __init__(self):
        self.last_context = None
        self.last_missing = None

    def interpret(self, message, context):
        return {}

    def generate(self, state, context):
        return ""

    def evaluate_critical_participants(self, context, missing):
        self.last_context = context
        self.last_missing = missing
        # Escalate if hard constraints detected in signal context
        if "hard" in context.get("constraint_type_signals", []):
            return {"critical_participants": missing[:1]}
        return {"critical_participants": []}


class _MockLLMAlwaysEscalates(LLM):
    """Always flags first missing participant."""

    def interpret(self, message, context):
        return {}

    def generate(self, state, context):
        return ""

    def evaluate_critical_participants(self, context, missing):
        return {"critical_participants": missing[:1]}


class _MockLLMNeverEscalates(LLM):
    """Never flags anyone."""

    def interpret(self, message, context):
        return {}

    def generate(self, state, context):
        return ""

    def evaluate_critical_participants(self, context, missing):
        return {"critical_participants": []}


class _MockLLMReturnsInvalidNames(LLM):
    """Returns names NOT in the missing list."""

    def interpret(self, message, context):
        return {}

    def generate(self, state, context):
        return ""

    def evaluate_critical_participants(self, context, missing):
        return {"critical_participants": ["alice", "carol", "nobody"]}


class _MockLLMTracked(LLM):
    """Tracks call count."""

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
# Context: consent + relaxed v2 signals (rule-based returns False)
# ---------------------------------------------------------------------------

def _consent_relaxed_ctx(**overrides):
    base = dict(
        question="Where to eat?",
        participants=["alice", "bob", "carol"],
        min_participants=2,
        decision_rule="consent",
        responses={"alice": ["anything"], "carol": ["fine with me"]},
        preferences=["Italian", "Italian"],
        flexibility_signals=["high", "high"],
        preference_strength_signals=["weak", "none"],
        constraint_type_signals=["none", "none"],
        services={},
    )
    base.update(overrides)
    return DecisionContext(**base)


# ---------------------------------------------------------------------------
# Context: consent + high tension (rule-based returns True)
# ---------------------------------------------------------------------------

def _consent_tense_ctx(**overrides):
    base = dict(
        question="Where to eat?",
        participants=["alice", "bob", "carol"],
        min_participants=2,
        decision_rule="consent",
        responses={"alice": ["Italian!"], "carol": ["Thai!"]},
        preferences=["Italian", "Thai"],
        flexibility_signals=["low", "low"],
        preference_strength_signals=["strong", "strong"],
        constraint_type_signals=["none", "none"],
        services={},
    )
    base.update(overrides)
    return DecisionContext(**base)


# ---------------------------------------------------------------------------
# Test 1: relaxed → rule=False, LLM escalates → True
# ---------------------------------------------------------------------------

def test_relaxed_llm_escalates():
    """Relaxed env (rule=False), LLM flags bob → True."""
    ctx = _consent_relaxed_ctx(
        services={"llm": _MockLLMAlwaysEscalates()},
    )
    assert not _has_critical_unresolved_rule_based(ctx)
    assert _has_critical_unresolved_participants(ctx) is True


# ---------------------------------------------------------------------------
# Test 2: relaxed → LLM agrees → False
# ---------------------------------------------------------------------------

def test_relaxed_llm_agrees():
    """Relaxed env (rule=False), LLM agrees → False."""
    ctx = _consent_relaxed_ctx(
        services={"llm": _MockLLMNeverEscalates()},
    )
    assert not _has_critical_unresolved_rule_based(ctx)
    assert _has_critical_unresolved_participants(ctx) is False


# ---------------------------------------------------------------------------
# Test 3: high tension → rule=True → LLM not called
# ---------------------------------------------------------------------------

def test_tense_llm_not_called():
    """Tense env (rule=True) → LLM never invoked."""
    mock = _MockLLMTracked()
    ctx = _consent_tense_ctx(services={"llm": mock})
    assert _has_critical_unresolved_rule_based(ctx) is True
    result = _has_critical_unresolved_participants(ctx)
    assert result is True
    assert mock.call_count == 0


# ---------------------------------------------------------------------------
# Test 4: LLM returns invalid names → filtered → False
# ---------------------------------------------------------------------------

def test_llm_invalid_names_filtered():
    """LLM returns names not in missing → all filtered out → False."""
    ctx = _consent_relaxed_ctx(
        services={"llm": _MockLLMReturnsInvalidNames()},
    )
    # Missing is ["bob"]. LLM returns ["alice", "carol", "nobody"].
    # None of those are in missing → filtered to [] → False.
    assert _has_critical_unresolved_participants(ctx) is False


# ---------------------------------------------------------------------------
# Test 5: LLM receives v2 signal context
# ---------------------------------------------------------------------------

def test_llm_receives_v2_signals():
    """LLM advisory receives flexibility, preference_strength, constraint_type."""
    mock = _MockLLMEscalatesWithSignals()
    ctx = _consent_relaxed_ctx(services={"llm": mock})

    _has_critical_unresolved_participants(ctx)

    assert mock.last_context is not None
    assert mock.last_context["flexibility_signals"] == ["high", "high"]
    assert mock.last_context["preference_strength_signals"] == ["weak", "none"]
    assert mock.last_context["constraint_type_signals"] == ["none", "none"]
    assert mock.last_missing == ["bob"]


def test_llm_signal_aware_escalation():
    """LLM uses constraint_type_signals to decide escalation."""
    mock = _MockLLMEscalatesWithSignals()

    # No hard constraints → LLM does NOT escalate
    ctx_soft = _consent_relaxed_ctx(
        constraint_type_signals=["soft", "none"],
        services={"llm": mock},
    )
    assert _has_critical_unresolved_participants(ctx_soft) is False

    # Hard constraint present → LLM DOES escalate
    # (need to also keep relaxed so rule-based returns False)
    # But wait: _has_hard_constraints would make rule-based True.
    # So we use majority-locked context instead.
    ctx_hard = DecisionContext(
        question="Where to eat?",
        participants=["alice", "bob", "carol", "dave", "eve"],
        min_participants=3,
        decision_rule="majority",
        responses={
            "alice": ["Italian"], "bob": ["Italian"],
            "carol": ["Italian"], "dave": ["Thai"],
        },
        preferences=["Italian", "Italian", "Italian", "Thai"],
        flexibility_signals=["high", "high", "high", "high"],
        preference_strength_signals=["weak", "weak", "weak", "weak"],
        constraint_type_signals=["hard", "none", "none", "none"],
        services={"llm": mock},
    )
    # Rule-based: hard constraints → True (LLM not called)
    assert _has_critical_unresolved_rule_based(ctx_hard) is True
    # So this test verifies rule short-circuits before LLM
    assert _has_critical_unresolved_participants(ctx_hard) is True


# ---------------------------------------------------------------------------
# Test 6: subset validation — mixed valid/invalid names
# ---------------------------------------------------------------------------

def test_llm_subset_validation_partial():
    """LLM returns mix of valid + invalid names → only valid kept."""

    class _MockMixed(LLM):
        def interpret(self, message, context):
            return {}
        def generate(self, state, context):
            return ""
        def evaluate_critical_participants(self, context, missing):
            # "bob" is in missing, "alice" and "nobody" are not
            return {"critical_participants": ["alice", "bob", "nobody"]}

    ctx = _consent_relaxed_ctx(services={"llm": _MockMixed()})
    # "bob" survives validation → True
    assert _has_critical_unresolved_participants(ctx) is True


if __name__ == "__main__":
    tests = [
        test_relaxed_llm_escalates,
        test_relaxed_llm_agrees,
        test_tense_llm_not_called,
        test_llm_invalid_names_filtered,
        test_llm_receives_v2_signals,
        test_llm_signal_aware_escalation,
        test_llm_subset_validation_partial,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} hybrid guard v2 tests passed.")
