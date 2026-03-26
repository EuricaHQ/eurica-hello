from enum import Enum


class Event(str, Enum):
    RESPONSE_RECEIVED = "response_received"
    AGGREGATION_COMPLETED = "aggregation_completed"
    CONFLICT_DETECTED = "conflict_detected"
    AVOIDANCE_DETECTED = "avoidance_detected"
    VALIDATION_REQUIRED = "validation_required"
    VALIDATION_COMPLETED = "validation_completed"
    DECISION_CONFIRMED = "decision_confirmed"
    DECISION_REJECTED = "decision_rejected"

    # NOTE: AGGREGATION_COMPLETED will later be split into finer-grained
    # events (e.g. solution_found, no_solution) as the aggregation
    # semantics evolve. Current usage is intentionally broad.
