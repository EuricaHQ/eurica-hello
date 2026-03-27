from enum import Enum


class Event(str, Enum):
    """Core event set per spec v2.9 section 22."""

    RESPONSE_RECEIVED = "response_received"
    AGGREGATION_COMPLETED = "aggregation_completed"
    VALIDATION_COMPLETED = "validation_completed"
    DECISION_CONFIRMED = "decision_confirmed"
    DECISION_REJECTED = "decision_rejected"
    AVOIDANCE_DETECTED = "avoidance_detected"
