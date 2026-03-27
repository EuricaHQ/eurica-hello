"""Tests for Signal Layer v2 (semantic signals).

Verifies:
- MockLLM produces correct flexibility / preference_strength / constraint_type
- OpenAI validation enforces schema (enum values, defaults)
- New signals don't break existing schema

Run: python3 test_signals_v2.py
"""

from llm.mock_llm import MockLLM
from llm.openai_llm import _SAFE_DEFAULT, _SIGNAL_SCHEMA_ENUMS, _SIGNAL_ENUM_DEFAULTS

_EXPECTED_KEYS = set(_SAFE_DEFAULT.keys())
_mock = MockLLM()
_ctx = {"question": "When should we meet?", "participants": [], "preferences": []}


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_safe_default_has_all_keys():
    """Safe default contains all v1 + v2 keys."""
    assert "flexibility" in _SAFE_DEFAULT
    assert "preference_strength" in _SAFE_DEFAULT
    assert "constraint_type" in _SAFE_DEFAULT
    assert _SAFE_DEFAULT["flexibility"] == "medium"
    assert _SAFE_DEFAULT["preference_strength"] == "none"
    assert _SAFE_DEFAULT["constraint_type"] == "none"


def test_mock_returns_all_keys():
    """MockLLM interpret() returns all v1 + v2 keys."""
    signals = _mock.interpret("hello", _ctx)
    missing = _EXPECTED_KEYS - set(signals.keys())
    extra = set(signals.keys()) - _EXPECTED_KEYS
    assert not missing, f"missing keys: {missing}"
    assert not extra, f"extra keys: {extra}"


# ---------------------------------------------------------------------------
# MockLLM: flexibility
# ---------------------------------------------------------------------------

def test_flexibility_high_egal():
    """'egal' → flexibility=high."""
    signals = _mock.interpret("mir ist das egal", _ctx)
    assert signals["flexibility"] == "high"


def test_flexibility_high_anything():
    """'anything' → flexibility=high."""
    signals = _mock.interpret("anything works for me", _ctx)
    assert signals["flexibility"] == "high"


def test_flexibility_low_will():
    """'will' → flexibility=low (strict preference)."""
    signals = _mock.interpret("ich will italienisch", _ctx)
    assert signals["flexibility"] == "low"


def test_flexibility_medium_default():
    """neutral input → flexibility=medium."""
    signals = _mock.interpret("ok", _ctx)
    assert signals["flexibility"] == "medium"


# ---------------------------------------------------------------------------
# MockLLM: preference_strength
# ---------------------------------------------------------------------------

def test_preference_strong_will():
    """'will' → preference_strength=strong."""
    signals = _mock.interpret("ich will italienisch", _ctx)
    assert signals["preference_strength"] == "strong"


def test_preference_strong_definitely():
    """'definitely' → preference_strength=strong."""
    signals = _mock.interpret("definitely Wednesday", _ctx)
    assert signals["preference_strength"] == "strong"


def test_preference_weak_vielleicht():
    """'vielleicht' → preference_strength=weak."""
    signals = _mock.interpret("vielleicht Donnerstag", _ctx)
    assert signals["preference_strength"] == "weak"


def test_preference_weak_maybe():
    """'maybe' → preference_strength=weak."""
    signals = _mock.interpret("maybe Thursday", _ctx)
    assert signals["preference_strength"] == "weak"


def test_preference_none_default():
    """neutral input → preference_strength=none."""
    signals = _mock.interpret("ok", _ctx)
    assert signals["preference_strength"] == "none"


# ---------------------------------------------------------------------------
# MockLLM: constraint_type
# ---------------------------------------------------------------------------

def test_constraint_hard_auf_keinen_fall():
    """'auf keinen fall' → constraint_type=hard."""
    signals = _mock.interpret("auf keinen fall mittwoch", _ctx)
    assert signals["constraint_type"] == "hard"


def test_constraint_hard_must():
    """'must' → constraint_type=hard."""
    signals = _mock.interpret("it must be after 5pm", _ctx)
    assert signals["constraint_type"] == "hard"


def test_constraint_soft_lieber_nicht():
    """'lieber nicht' → constraint_type=soft."""
    signals = _mock.interpret("lieber nicht am montag", _ctx)
    assert signals["constraint_type"] == "soft"


def test_constraint_none_default():
    """neutral input → constraint_type=none."""
    signals = _mock.interpret("ok", _ctx)
    assert signals["constraint_type"] == "none"


# ---------------------------------------------------------------------------
# Defaults applied correctly
# ---------------------------------------------------------------------------

def test_defaults_on_neutral_input():
    """Neutral input gets all defaults correct."""
    signals = _mock.interpret("ok", _ctx)
    assert signals["flexibility"] == "medium"
    assert signals["preference_strength"] == "none"
    assert signals["constraint_type"] == "none"
    assert signals["uncertainty"] is False
    assert signals["avoidance"] is False


# ---------------------------------------------------------------------------
# V1 signals still work
# ---------------------------------------------------------------------------

def test_v1_avoidance_still_works():
    """'later' still triggers avoidance=True."""
    signals = _mock.interpret("maybe later", _ctx)
    assert signals["avoidance"] is True


def test_v1_fields_present():
    """All v1 fields are present and correctly typed."""
    signals = _mock.interpret("hello", _ctx)
    assert isinstance(signals["preferences"], list)
    assert isinstance(signals["constraints"], list)
    assert isinstance(signals["uncertainty"], bool)
    assert isinstance(signals["conflict"], bool)
    assert isinstance(signals["objection"], bool)
    assert isinstance(signals["avoidance"], bool)


# ---------------------------------------------------------------------------
# Validation: enum enforcement
# ---------------------------------------------------------------------------

def test_enum_defaults_match_schema():
    """All enum defaults are valid enum values."""
    for key, allowed in _SIGNAL_SCHEMA_ENUMS.items():
        default = _SIGNAL_ENUM_DEFAULTS[key]
        assert default in allowed, f"default '{default}' for {key} not in {allowed}"


if __name__ == "__main__":
    tests = [
        test_safe_default_has_all_keys,
        test_mock_returns_all_keys,
        # flexibility
        test_flexibility_high_egal,
        test_flexibility_high_anything,
        test_flexibility_low_will,
        test_flexibility_medium_default,
        # preference_strength
        test_preference_strong_will,
        test_preference_strong_definitely,
        test_preference_weak_vielleicht,
        test_preference_weak_maybe,
        test_preference_none_default,
        # constraint_type
        test_constraint_hard_auf_keinen_fall,
        test_constraint_hard_must,
        test_constraint_soft_lieber_nicht,
        test_constraint_none_default,
        # defaults
        test_defaults_on_neutral_input,
        # v1 compat
        test_v1_avoidance_still_works,
        test_v1_fields_present,
        # validation
        test_enum_defaults_match_schema,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} signal v2 tests passed.")
