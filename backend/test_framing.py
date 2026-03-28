"""Tests for framing flow — spec v2.11.0.

Covers:
- Framing trigger conditions
- At-most-once execution
- Non-blocking behavior (DECIDING still reachable)
- Initiator response mapping → expected_dimensions
- proposed_dimensions never auto-promoted
- clarify_dimension only after expected_dimensions defined

Run: python3 test_framing.py
"""

import unittest
from dataclasses import replace

from machine.states import State
from machine.events import Event
from machine.context import DecisionContext
from machine.transition import transition
from targeting import (
    should_trigger_framing,
    build_framing_payload,
    mark_framing_executed,
    map_framing_response,
)


def _base_context(**overrides) -> DecisionContext:
    defaults = dict(
        decision_id="framing",
        question="When should we meet?",
        participants=["alice"],
        min_participants=1,
        initiator="alice",
        responses={"alice": ["Wednesday"]},
        preferences=[{"value": "Wednesday", "dimension": "day"}],
    )
    defaults.update(overrides)
    return DecisionContext(**defaults)


class TestFramingTrigger(unittest.TestCase):

    def test_trigger_when_proposed_and_no_expected(self):
        """Framing triggers: proposed >= 2, expected=None, not executed."""
        ctx = _base_context(
            proposed_dimensions=["day", "time"],
            expected_dimensions=None,
            framing_executed=False,
        )
        self.assertTrue(should_trigger_framing(ctx))

    def test_no_trigger_when_expected_already_set(self):
        """Framing does not trigger when expected_dimensions exists."""
        ctx = _base_context(
            proposed_dimensions=["day", "time"],
            expected_dimensions=["day", "time"],
            framing_executed=False,
        )
        self.assertFalse(should_trigger_framing(ctx))

    def test_no_trigger_when_already_executed(self):
        """Framing executes at most once."""
        ctx = _base_context(
            proposed_dimensions=["day", "time"],
            expected_dimensions=None,
            framing_executed=True,
        )
        self.assertFalse(should_trigger_framing(ctx))

    def test_no_trigger_when_proposed_too_few(self):
        """Framing requires >= 2 proposed dimensions."""
        ctx = _base_context(
            proposed_dimensions=["day"],
            expected_dimensions=None,
        )
        self.assertFalse(should_trigger_framing(ctx))

    def test_no_trigger_when_proposed_none(self):
        """Framing requires proposed_dimensions to exist."""
        ctx = _base_context(
            proposed_dimensions=None,
            expected_dimensions=None,
        )
        self.assertFalse(should_trigger_framing(ctx))

    def test_no_trigger_when_proposed_empty(self):
        """Framing requires proposed_dimensions to be non-empty."""
        ctx = _base_context(
            proposed_dimensions=[],
            expected_dimensions=None,
        )
        self.assertFalse(should_trigger_framing(ctx))


class TestFramingPayload(unittest.TestCase):

    def test_payload_structure(self):
        """Payload contains interaction_type, target, proposed_dimensions."""
        ctx = _base_context(
            proposed_dimensions=["day", "time", "cuisine"],
            initiator="alice",
        )
        payload = build_framing_payload(ctx)
        self.assertEqual(payload["interaction_type"], "framing")
        self.assertEqual(payload["target"], "alice")
        self.assertEqual(payload["proposed_dimensions"], ["day", "time", "cuisine"])

    def test_payload_target_falls_back_to_first_participant(self):
        """If no initiator, target is first participant."""
        ctx = _base_context(
            proposed_dimensions=["day", "time"],
            initiator=None,
        )
        payload = build_framing_payload(ctx)
        self.assertEqual(payload["target"], "alice")


class TestFramingExecution(unittest.TestCase):

    def test_mark_executed(self):
        """mark_framing_executed sets flag."""
        ctx = _base_context(framing_executed=False)
        updated = mark_framing_executed(ctx)
        self.assertTrue(updated.framing_executed)

    def test_mark_executed_idempotent(self):
        """Marking already-executed context is safe."""
        ctx = _base_context(framing_executed=True)
        updated = mark_framing_executed(ctx)
        self.assertTrue(updated.framing_executed)


class TestFramingResponseMapping(unittest.TestCase):

    def test_all_included(self):
        """All dimensions explicitly included."""
        result = map_framing_response([
            {"dimension": "day", "status": "include"},
            {"dimension": "time", "status": "include"},
        ])
        self.assertEqual(sorted(result), ["day", "time"])

    def test_some_excluded(self):
        """Excluded dimensions are omitted."""
        result = map_framing_response([
            {"dimension": "day", "status": "include"},
            {"dimension": "time", "status": "exclude"},
            {"dimension": "cuisine", "status": "include"},
        ])
        self.assertEqual(sorted(result), ["cuisine", "day"])

    def test_neutral_returns_none(self):
        """Any neutral dimension → entire result is None."""
        result = map_framing_response([
            {"dimension": "day", "status": "include"},
            {"dimension": "time", "status": "neutral"},
        ])
        self.assertIsNone(result)

    def test_empty_input_returns_none(self):
        """Empty response → None."""
        result = map_framing_response([])
        self.assertIsNone(result)

    def test_all_excluded_returns_none(self):
        """All excluded → no included dimensions → None."""
        result = map_framing_response([
            {"dimension": "day", "status": "exclude"},
            {"dimension": "time", "status": "exclude"},
        ])
        self.assertIsNone(result)

    def test_unknown_status_returns_none(self):
        """Unknown status treated as neutral → None."""
        result = map_framing_response([
            {"dimension": "day", "status": "include"},
            {"dimension": "time", "status": "maybe"},
        ])
        self.assertIsNone(result)


class TestFramingNonBlocking(unittest.TestCase):

    def test_deciding_reachable_without_framing(self):
        """Framing not executed + expected=None → DECIDING still reachable."""
        ctx = _base_context(
            proposed_dimensions=["day", "time", "cuisine"],
            expected_dimensions=None,
            framing_executed=False,
        )
        next_state, _, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        # expected_dimensions=None → completeness satisfied → DECIDING
        self.assertEqual(next_state, State.DECIDING)

    def test_deciding_reachable_after_framing_no_response(self):
        """Framing executed but no response → expected stays None → DECIDING."""
        ctx = _base_context(
            proposed_dimensions=["day", "time"],
            expected_dimensions=None,
            framing_executed=True,
        )
        next_state, _, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(next_state, State.DECIDING)


class TestProposedNeverAutoPromoted(unittest.TestCase):

    def test_proposed_not_used_as_expected(self):
        """proposed_dimensions must never be treated as expected_dimensions."""
        ctx = _base_context(
            proposed_dimensions=["day", "time"],
            expected_dimensions=None,
        )
        # Even with proposed set, expected=None → no dimension incompleteness
        next_state, _, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx,
        )
        self.assertEqual(next_state, State.DECIDING)


class TestClarifyDimensionConstraint(unittest.TestCase):

    def test_clarify_dimension_only_with_expected(self):
        """clarify_dimension requires expected_dimensions to be defined."""
        # With expected + missing "time" → incomplete → clarify_dimension
        ctx1 = _base_context(expected_dimensions=["day", "time"])
        next_state1, actions1, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx1,
        )
        self.assertEqual(next_state1, State.AGGREGATING)
        self.assertIsNotNone(actions1[0].payload)
        self.assertEqual(actions1[0].payload["interaction_type"], "clarify_dimension")

        # Without expected → DECIDING (no clarify_dimension)
        ctx2 = _base_context(expected_dimensions=None)
        next_state, actions2, _ = transition(
            State.AGGREGATING, Event.AGGREGATION_COMPLETED, ctx2,
        )
        self.assertEqual(next_state, State.DECIDING)
        self.assertIsNone(actions2[0].payload)

    def test_framing_does_not_trigger_clarify_dimension(self):
        """Framing interaction_type is 'framing', not 'clarify_dimension'."""
        ctx = _base_context(
            proposed_dimensions=["day", "time"],
            expected_dimensions=None,
        )
        payload = build_framing_payload(ctx)
        self.assertEqual(payload["interaction_type"], "framing")
        self.assertNotEqual(payload["interaction_type"], "clarify_dimension")


if __name__ == "__main__":
    unittest.main()
