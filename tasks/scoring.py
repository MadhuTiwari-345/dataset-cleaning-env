SCORE_EPSILON = 0.001


def clamp_open_unit_interval(value: float, epsilon: float = SCORE_EPSILON) -> float:
    """Clamp a score so it is always strictly inside the open unit interval."""
    if epsilon <= 0.0 or epsilon >= 0.5:
        raise ValueError("epsilon must be between 0.0 and 0.5")
    return min(max(float(value), epsilon), 1.0 - epsilon)
