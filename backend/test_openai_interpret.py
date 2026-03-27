"""Integration test: OpenAI LLM interpret() with real API calls.

Verifies:
- No crashes on real API roundtrip
- Output is a dict (parsed from JSON internally)
- Schema has EXACTLY the 6 allowed keys
- List fields contain lists, bool fields contain bools
- No extra keys leak through

Requires: OPENAI_API_KEY in environment or .env

Run: python3 test_openai_interpret.py
"""

from dotenv import load_dotenv

load_dotenv()

from llm.openai_llm import OpenAILLM

_EXPECTED_LIST_KEYS = {"preferences", "constraints"}
_EXPECTED_BOOL_KEYS = {"uncertainty", "conflict", "objection", "avoidance"}
_EXPECTED_KEYS = _EXPECTED_LIST_KEYS | _EXPECTED_BOOL_KEYS

_CONTEXT = {
    "question": "When should we meet this week?",
    "participants": ["alice", "bob"],
    "preferences": ["Tuesday"],
}


def _assert_schema(signals: dict, label: str):
    """Assert strict schema compliance on a signals dict."""

    # Must be a dict
    assert isinstance(signals, dict), (
        f"[{label}] expected dict, got {type(signals).__name__}"
    )

    # Exactly the allowed keys — no more, no fewer
    assert set(signals.keys()) == _EXPECTED_KEYS, (
        f"[{label}] key mismatch: "
        f"extra={set(signals.keys()) - _EXPECTED_KEYS}, "
        f"missing={_EXPECTED_KEYS - set(signals.keys())}"
    )

    # List fields must be lists
    for key in _EXPECTED_LIST_KEYS:
        assert isinstance(signals[key], list), (
            f"[{label}] {key}: expected list, got {type(signals[key]).__name__}"
        )

    # Bool fields must be bools
    for key in _EXPECTED_BOOL_KEYS:
        assert isinstance(signals[key], bool), (
            f"[{label}] {key}: expected bool, got {type(signals[key]).__name__}"
        )


def test_preference_message():
    """'I can Wednesday or Thursday' → valid schema, preferences populated."""
    llm = OpenAILLM()
    signals = llm.interpret("I can Wednesday or Thursday", _CONTEXT)
    _assert_schema(signals, "preference")
    print(f"  signals: {signals}")


def test_uncertainty_message():
    """'I am not sure' → valid schema, uncertainty should be true."""
    llm = OpenAILLM()
    signals = llm.interpret("I am not sure", _CONTEXT)
    _assert_schema(signals, "uncertainty")
    assert signals["uncertainty"] is True, (
        f"expected uncertainty=True, got {signals['uncertainty']}"
    )
    # Uncertainty is NOT a constraint
    assert signals["constraints"] == [], (
        f"'I am not sure' must not produce constraints, got {signals['constraints']}"
    )
    print(f"  signals: {signals}")


def test_avoidance_message():
    """'maybe later' → valid schema, avoidance should be true."""
    llm = OpenAILLM()
    signals = llm.interpret("maybe later", _CONTEXT)
    _assert_schema(signals, "avoidance")
    assert signals["avoidance"] is True, (
        f"expected avoidance=True, got {signals['avoidance']}"
    )
    print(f"  signals: {signals}")


if __name__ == "__main__":
    tests = [
        test_preference_message,
        test_uncertainty_message,
        test_avoidance_message,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}\n")
    print(f"{len(tests)}/{len(tests)} integration tests passed.")
