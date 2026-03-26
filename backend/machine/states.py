from enum import Enum


class State(str, Enum):
    CLARIFYING = "clarifying"
    COLLECTING = "collecting"
    AGGREGATING = "aggregating"
    RESOLVING = "resolving"
    AVOIDING = "avoiding"
    VALIDATING = "validating"
    DECIDING = "deciding"
    DECIDED = "decided"
    INFEASIBLE = "infeasible"
