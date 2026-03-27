from __future__ import annotations

from llm.interface import LLM


class MockLLM(LLM):
    """Mock LLM for development and testing.

    interpret(): keyword-based signal extraction (no real LLM).
    generate(): template-based response (no real LLM).
    """

    def interpret(self, message: str, context: dict) -> dict:
        lower = message.lower()

        # Keyword-based flexibility detection
        if any(w in lower for w in ("egal", "anything", "passt alles")):
            flexibility = "high"
        elif any(w in lower for w in ("will", "must", "definitely", "auf jeden")):
            flexibility = "low"
        else:
            flexibility = "medium"

        # Keyword-based preference strength
        if any(w in lower for w in ("will", "definitely", "auf jeden", "unbedingt")):
            preference_strength = "strong"
        elif any(w in lower for w in ("vielleicht", "maybe", "could", "might")):
            preference_strength = "weak"
        else:
            preference_strength = "none"

        # Keyword-based constraint type
        if any(w in lower for w in ("muss", "must", "cannot", "auf keinen fall", "niemals")):
            constraint_type = "hard"
        elif any(w in lower for w in ("lieber nicht", "prefer not", "rather not")):
            constraint_type = "soft"
        else:
            constraint_type = "none"

        return {
            "preferences": [],
            "constraints": [],
            "uncertainty": False,
            "conflict": False,
            "objection": False,
            "avoidance": "later" in lower,
            "flexibility": flexibility,
            "preference_strength": preference_strength,
            "constraint_type": constraint_type,
        }

    def generate(self, state: str, context: dict) -> str:
        return f"[{state}] What is your preference?"

    def evaluate_critical_participants(
        self, context: dict, missing: list[str],
    ) -> dict:
        return {"critical_participants": []}
