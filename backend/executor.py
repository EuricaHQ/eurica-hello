from __future__ import annotations

from machine.actions import Action, ActionType
from machine.context import DecisionContext


def execute(actions: list[Action], context: DecisionContext) -> str:
    """Execute actions and return a response string.

    Each action type maps to a handler. Handlers are minimal stubs
    that will be replaced with real logic later.

    No external calls. No side effects beyond building a reply.
    """
    if not actions:
        return "No action required."

    parts: list[str] = []
    for action in actions:
        handler = _HANDLERS.get(action.type, _default_handler)
        parts.append(handler(action, context))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Action handlers — each produces a reply fragment
# ---------------------------------------------------------------------------

def _ask_question(action: Action, ctx: DecisionContext) -> str:
    if not ctx.question:
        return "What decision do you need help with?"
    # Build a contextual follow-up
    n_responses = sum(len(r) for r in ctx.responses.values())
    if n_responses == 0:
        return f"Regarding '{ctx.question}' — what are your preferences or constraints?"
    return f"Thanks. Anything else to add about '{ctx.question}'?"


def _aggregate(action: Action, ctx: DecisionContext) -> str:
    all_inputs = []
    for participant, responses in ctx.responses.items():
        all_inputs.extend(responses)
    summary = "; ".join(all_inputs) if all_inputs else "no inputs"
    return f"[Aggregating] Inputs so far: {summary}"


def _validate_constraint(action: Action, ctx: DecisionContext) -> str:
    constraints = ", ".join(ctx.constraints) if ctx.constraints else "none"
    return f"[Validating] Checking constraints: {constraints}"


def _resolve_conflict(action: Action, ctx: DecisionContext) -> str:
    return "[Resolving] There are conflicting inputs. Can you help clarify?"


def _propose_decision(action: Action, ctx: DecisionContext) -> str:
    all_prefs = ", ".join(ctx.preferences) if ctx.preferences else "your inputs"
    return f"[Proposing] Based on {all_prefs}, here is a proposed decision. Do you confirm?"


def _finalize_decision(action: Action, ctx: DecisionContext) -> str:
    return f"[Decided] Decision finalized: {ctx.decision or 'confirmed'}"


def _mark_infeasible(action: Action, ctx: DecisionContext) -> str:
    return "[Infeasible] No viable solution found given the current constraints."


def _address_avoidance(action: Action, ctx: DecisionContext) -> str:
    return "[Avoidance detected] It seems like this decision is being avoided. Let's refocus."


def _default_handler(action: Action, ctx: DecisionContext) -> str:
    return f"[{action.type.value}] Action acknowledged."


_HANDLERS: dict[ActionType, callable] = {
    ActionType.ASK_QUESTION: _ask_question,
    ActionType.AGGREGATE: _aggregate,
    ActionType.VALIDATE_CONSTRAINT: _validate_constraint,
    ActionType.RESOLVE_CONFLICT: _resolve_conflict,
    ActionType.PROPOSE_DECISION: _propose_decision,
    ActionType.FINALIZE_DECISION: _finalize_decision,
    ActionType.MARK_INFEASIBLE: _mark_infeasible,
    ActionType.ADDRESS_AVOIDANCE: _address_avoidance,
}
