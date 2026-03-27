"""End-to-end smoke test: real OpenAI calls through the full LLM layer.

Exercises the complete pipeline per step:
  interpret() → signal_mapper → transition (+ system events) → generate()

No mocks. Real API calls.

Requires: OPENAI_API_KEY in environment or .env

Run: python3 test_e2e_smoke.py
"""

from dotenv import load_dotenv

load_dotenv()

from dataclasses import replace

from llm.openai_llm import OpenAILLM
from interpreter.signal_mapper import map_signals_to_event
from machine.states import State
from machine.events import Event
from machine.context import DecisionContext
from machine.transition import transition
from api.routes import _derive_system_event

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

_EXPECTED_LIST_KEYS = {"preferences", "constraints"}
_EXPECTED_BOOL_KEYS = {"uncertainty", "conflict", "objection", "avoidance"}
_EXPECTED_KEYS = _EXPECTED_LIST_KEYS | _EXPECTED_BOOL_KEYS


def _validate_schema(signals: dict, step: int):
    assert isinstance(signals, dict), (
        f"Step {step}: expected dict, got {type(signals).__name__}"
    )
    extra = set(signals.keys()) - _EXPECTED_KEYS
    missing = _EXPECTED_KEYS - set(signals.keys())
    assert not extra, f"Step {step}: extra keys: {extra}"
    assert not missing, f"Step {step}: missing keys: {missing}"
    for k in _EXPECTED_LIST_KEYS:
        assert isinstance(signals[k], list), (
            f"Step {step}: {k} must be list, got {type(signals[k]).__name__}"
        )
    for k in _EXPECTED_BOOL_KEYS:
        assert isinstance(signals[k], bool), (
            f"Step {step}: {k} must be bool, got {type(signals[k]).__name__}"
        )


# ---------------------------------------------------------------------------
# System event loop (mirrors routes.py logic, no route dependency)
# ---------------------------------------------------------------------------

_MAX_SYSTEM_ITERATIONS = 3


def _run_system_events(state, ctx):
    """Run system event loop after user-driven transition."""
    for _ in range(_MAX_SYSTEM_ITERATIONS):
        sys_event = _derive_system_event(state, ctx)
        if sys_event is None:
            break
        prev = state
        state, actions, ctx = transition(state, sys_event, ctx)
        if state == prev:
            break
    return state, ctx


# ---------------------------------------------------------------------------
# Context helper
# ---------------------------------------------------------------------------

def _context_to_dict(ctx: DecisionContext) -> dict:
    return {
        "question": ctx.question,
        "participants": ctx.participants,
        "preferences": ctx.preferences,
        "constraints": ctx.constraints,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    llm = OpenAILLM()

    ctx = DecisionContext(
        decision_id="smoke",
        question="When should we meet?",
        participants=["alice", "bob"],
        min_participants=1,
        responses={"alice": ["Tuesday"]},
        preferences=[],
        constraints=[],
    )

    state = State.COLLECTING

    inputs = [
        "I can Wednesday or Thursday",
        "I am not sure yet",
        "maybe later",
    ]

    failed = False

    for i, message in enumerate(inputs, 1):
        step = i
        state_before = state

        print(f"\n{'=' * 60}")
        print(f"--- STEP {step} ---")
        print(f"{'=' * 60}")

        # --- interpret ---
        print(f"\nINPUT: {message!r}")
        signals = llm.interpret(message, _context_to_dict(ctx))
        print(f"SIGNALS: {signals}")

        # --- schema validation ---
        try:
            _validate_schema(signals, step)
            print("SCHEMA: OK")
        except AssertionError as e:
            print(f"SCHEMA: FAIL — {e}")
            failed = True
            continue

        # --- semantic checks ---
        if step == 1:
            if len(signals["preferences"]) == 0:
                print("SEMANTIC: WARN — expected preferences for availability input")
            if signals["uncertainty"] is True:
                print("SEMANTIC: FAIL — uncertainty should be false")
                failed = True
            else:
                print("SEMANTIC: OK (no false uncertainty)")

        if step == 2:
            if signals["uncertainty"] is not True:
                print(f"SEMANTIC: FAIL — expected uncertainty=True, got {signals['uncertainty']}")
                failed = True
            elif len(signals["constraints"]) > 0:
                print(f"SEMANTIC: FAIL — 'not sure' is NOT a constraint, got {signals['constraints']}")
                failed = True
            else:
                print("SEMANTIC: OK (uncertainty=True, no false constraints)")

        if step == 3:
            if signals["avoidance"] is not True:
                print(f"SEMANTIC: FAIL — expected avoidance=True, got {signals['avoidance']}")
                failed = True
            else:
                print("SEMANTIC: OK (avoidance=True)")

        # --- signal mapping ---
        event = map_signals_to_event(signals)
        print(f"EVENT: {event.value}")

        # --- merge signals into context ---
        ctx = replace(
            ctx,
            preferences=ctx.preferences + signals.get("preferences", []),
            constraints=ctx.constraints + signals.get("constraints", []),
            uncertainties=ctx.uncertainties + (
                ["uncertainty"] if signals.get("uncertainty") else []
            ),
            objections=ctx.objections + (
                ["objection"] if signals.get("objection") else []
            ),
        )

        # --- transition ---
        next_state, actions, ctx = transition(state, event, ctx)
        print(f"STATE BEFORE: {state_before.value}")

        # --- system events ---
        next_state, ctx = _run_system_events(next_state, ctx)
        print(f"STATE AFTER: {next_state.value}")

        # --- guard: no direct jump to DECIDING ---
        if state_before == State.COLLECTING and next_state == State.DECIDING:
            print("TRANSITION: FAIL — direct jump from COLLECTING to DECIDING")
            failed = True
        else:
            print(f"TRANSITION: OK")

        state = next_state

        # --- generate ---
        response = llm.generate(state.value, _context_to_dict(ctx))
        print(f"RESPONSE: {response}")

        # --- response sanity ---
        if not response or len(response) > 1000:
            print(f"RESPONSE CHECK: FAIL — empty or too long ({len(response)} chars)")
            failed = True
        else:
            print(f"RESPONSE CHECK: OK ({len(response)} chars)")

    # --- summary ---
    print(f"\n{'=' * 60}")
    if failed:
        print("RESULT: FAIL — see errors above")
        raise SystemExit(1)
    else:
        print("RESULT: ALL STEPS PASSED")


if __name__ == "__main__":
    main()
