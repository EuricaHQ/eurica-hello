from __future__ import annotations

from interpreter.signals import Signals


def interpret_message(message: str) -> Signals:
    """Call LLM to extract structured signals from a user message.

    Returns Signals, NOT Events. The signal-to-event mapping
    is handled by signals.map_signals_to_event().

    TODO: Implement actual LLM call (OpenAI). Currently a stub.
    """
    return Signals()
