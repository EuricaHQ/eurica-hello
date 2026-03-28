"""Targeting Layer — spec v2.11.0 framing flow.

Pure functions. No transition logic. No guard modifications.
Framing connects proposed_dimensions → expected_dimensions via initiator.
"""

from __future__ import annotations

from dataclasses import replace

from machine.context import DecisionContext


def should_trigger_framing(ctx: DecisionContext) -> bool:
    """Spec v2.11.0: evaluate whether framing should be triggered.

    All conditions must hold:
    - proposed_dimensions exists with >= 2 entries
    - expected_dimensions is not yet set
    - framing has not already been executed
    """
    if ctx.framing_executed:
        return False
    if ctx.expected_dimensions is not None:
        return False
    if not ctx.proposed_dimensions or len(ctx.proposed_dimensions) < 2:
        return False
    return True


def build_framing_payload(ctx: DecisionContext) -> dict:
    """Build the payload for a framing interaction.

    Returns a dict suitable for passing to the LLM generate layer.
    The LLM produces the actual phrasing — no hardcoded text here.
    """
    return {
        "interaction_type": "framing",
        "target": ctx.initiator or (ctx.participants[0] if ctx.participants else None),
        "proposed_dimensions": list(ctx.proposed_dimensions or []),
    }


def mark_framing_executed(ctx: DecisionContext) -> DecisionContext:
    """Mark framing as executed (at-most-once)."""
    return replace(ctx, framing_executed=True)


def map_framing_response(
    response_dimensions: list[dict],
) -> list[str] | None:
    """Spec v2.11.0 §6: map initiator framing response → expected_dimensions.

    Each entry in response_dimensions should be:
      {"dimension": str, "status": "include" | "exclude" | "neutral"}

    Rules:
    - "include" → add to expected_dimensions
    - "exclude" → omit
    - "neutral" / unknown → if ANY dimension is neutral, return None
      (unclear response → no expected_dimensions set)

    Returns:
    - list[str] of included dimensions, or
    - None if response is unclear/neutral
    """
    if not response_dimensions:
        return None

    included: list[str] = []
    for entry in response_dimensions:
        status = entry.get("status", "neutral")
        dimension = entry.get("dimension", "")
        if status == "neutral" or status not in ("include", "exclude"):
            return None
        if status == "include" and dimension:
            included.append(dimension)

    return included if included else None
