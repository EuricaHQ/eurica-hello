from __future__ import annotations

from typing import Callable

from dataclasses import replace

from machine.states import State
from machine.events import Event
from machine.context import DecisionContext
from machine.actions import Action, ActionType


# Type aliases
Guard = Callable[[DecisionContext], bool]
TransitionResult = tuple[State, list[Action], DecisionContext]

# Terminal states — no transitions allowed out of these
_TERMINAL_STATES: frozenset[State] = frozenset({State.DECIDED, State.INFEASIBLE})


# ---------------------------------------------------------------------------
# Guards — pure predicates on context, no side effects
# ---------------------------------------------------------------------------

def _always(_ctx: DecisionContext) -> bool:
    return True


def _participation_satisfied(ctx: DecisionContext) -> bool:
    """Check if minimum participation requirement is met.

    Counts participants who have submitted at least one response.
    """
    return len(ctx.responses) >= ctx.min_participants


def _participation_not_satisfied(ctx: DecisionContext) -> bool:
    return not _participation_satisfied(ctx)


def _participation_satisfied_and_no_conflicts(ctx: DecisionContext) -> bool:
    return _participation_satisfied(ctx) and not _has_conflicts(ctx)


def _has_conflicts(ctx: DecisionContext) -> bool:
    return len(ctx.conflicts) > 0


def _no_conflicts(ctx: DecisionContext) -> bool:
    return not _has_conflicts(ctx)


def _has_validation_errors(ctx: DecisionContext) -> bool:
    return len(ctx.validation_errors) > 0


def _no_validation_errors(ctx: DecisionContext) -> bool:
    return not _has_validation_errors(ctx)


def _participation_satisfied_and_no_validation_errors(ctx: DecisionContext) -> bool:
    return _participation_satisfied(ctx) and not _has_validation_errors(ctx)


def _no_solution(ctx: DecisionContext) -> bool:
    """Stub guard: true when aggregation found no viable solution.

    Will be refined when AGGREGATION_COMPLETED is split into
    finer-grained events (solution_found, no_solution, etc.).
    """
    return False


# ---------------------------------------------------------------------------
# Transition table
#
# Each entry: (state, event, guard) → (next_state, actions)
#
# Guards are evaluated top-to-bottom. First match wins.
# Handlers do NOT contain control flow — only the table decides.
# ---------------------------------------------------------------------------

_TransitionEntry = tuple[State, Event, Guard, State, list[ActionType]]

_TRANSITION_TABLE: list[_TransitionEntry] = [
    # --- CLARIFYING ---
    (State.CLARIFYING, Event.RESPONSE_RECEIVED, _always,
     State.COLLECTING, [ActionType.ASK_QUESTION]),

    # --- COLLECTING ---
    (State.COLLECTING, Event.RESPONSE_RECEIVED, _participation_satisfied,
     State.AGGREGATING, [ActionType.AGGREGATE]),

    (State.COLLECTING, Event.RESPONSE_RECEIVED, _participation_not_satisfied,
     State.COLLECTING, [ActionType.ASK_QUESTION]),

    # --- AGGREGATING (structural hub) ---
    # Aggregating is the central routing state. All paths from
    # collecting lead here, and it fans out to resolving, validating,
    # avoiding, deciding, or infeasible based on guards and events.
    #
    # NOTE: AGGREGATION_COMPLETED currently serves as a broad event.
    # It will later be split into finer-grained events (solution_found,
    # no_solution, etc.) as aggregation semantics evolve.

    (State.AGGREGATING, Event.CONFLICT_DETECTED, _always,
     State.RESOLVING, [ActionType.RESOLVE_CONFLICT]),

    (State.AGGREGATING, Event.AVOIDANCE_DETECTED, _always,
     State.AVOIDING, [ActionType.ADDRESS_AVOIDANCE]),

    (State.AGGREGATING, Event.VALIDATION_REQUIRED, _always,
     State.VALIDATING, [ActionType.VALIDATE_CONSTRAINT]),

    (State.AGGREGATING, Event.AGGREGATION_COMPLETED, _no_solution,
     State.INFEASIBLE, [ActionType.MARK_INFEASIBLE]),

    (State.AGGREGATING, Event.AGGREGATION_COMPLETED, _has_conflicts,
     State.RESOLVING, [ActionType.RESOLVE_CONFLICT]),

    (State.AGGREGATING, Event.AGGREGATION_COMPLETED, _participation_satisfied_and_no_conflicts,
     State.DECIDING, [ActionType.PROPOSE_DECISION]),

    (State.AGGREGATING, Event.AGGREGATION_COMPLETED, _participation_not_satisfied,
     State.COLLECTING, [ActionType.ASK_QUESTION]),

    # --- RESOLVING ---
    (State.RESOLVING, Event.RESPONSE_RECEIVED, _has_conflicts,
     State.RESOLVING, [ActionType.RESOLVE_CONFLICT]),

    (State.RESOLVING, Event.RESPONSE_RECEIVED, _no_conflicts,
     State.AGGREGATING, [ActionType.AGGREGATE]),

    # --- VALIDATING ---
    (State.VALIDATING, Event.VALIDATION_COMPLETED, _no_solution,
     State.INFEASIBLE, [ActionType.MARK_INFEASIBLE]),

    (State.VALIDATING, Event.VALIDATION_COMPLETED, _has_validation_errors,
     State.RESOLVING, [ActionType.RESOLVE_CONFLICT]),

    (State.VALIDATING, Event.VALIDATION_COMPLETED, _participation_satisfied_and_no_validation_errors,
     State.DECIDING, [ActionType.PROPOSE_DECISION]),

    (State.VALIDATING, Event.VALIDATION_COMPLETED, _participation_not_satisfied,
     State.COLLECTING, [ActionType.ASK_QUESTION]),

    # --- DECIDING ---
    (State.DECIDING, Event.VALIDATION_REQUIRED, _always,
     State.VALIDATING, [ActionType.VALIDATE_CONSTRAINT]),

    (State.DECIDING, Event.CONFLICT_DETECTED, _always,
     State.RESOLVING, [ActionType.RESOLVE_CONFLICT]),

    (State.DECIDING, Event.DECISION_CONFIRMED, _always,
     State.DECIDED, [ActionType.FINALIZE_DECISION]),

    (State.DECIDING, Event.DECISION_REJECTED, _always,
     State.COLLECTING, [ActionType.ASK_QUESTION]),

    # --- AVOIDING ---
    (State.AVOIDING, Event.RESPONSE_RECEIVED, _always,
     State.COLLECTING, [ActionType.ASK_QUESTION]),
]


# ---------------------------------------------------------------------------
# Central transition function
# ---------------------------------------------------------------------------

def transition(
    state: State,
    event: Event,
    context: DecisionContext,
) -> TransitionResult:
    """Central transition function.

    (state, event, context) → (next_state, actions, updated_context)

    ALL control flow goes through here. No exceptions.
    This function is pure: no I/O, no LLM calls, no external systems.

    Evaluation order:
    1. Terminal state check
    2. Walk table top-to-bottom
    3. First matching (state, event, guard) wins
    4. No match → stay in current state
    """
    # Terminal states are absorbing — no exit transitions
    if state in _TERMINAL_STATES:
        return (state, [], context)

    # Walk the table, first matching guard wins
    for t_state, t_event, guard, next_state, action_types in _TRANSITION_TABLE:
        if t_state == state and t_event == event and guard(context):
            actions = [Action(at) for at in action_types]
            return (next_state, actions, context)

    # No valid transition — stay in current state, no actions
    return (state, [], context)
