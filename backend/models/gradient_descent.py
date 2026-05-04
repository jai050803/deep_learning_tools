def gd_loss(weight: float, target: float) -> float:
    return (weight - target) ** 2


def gd_gradient(weight: float, target: float) -> float:
    return 2 * (weight - target)


def classify_convergence(diverged: bool, initial_loss: float, final_loss: float, trace_length: int) -> str:
    if diverged or final_loss > initial_loss * 2:
        return "diverging"
    if final_loss < 0.001:
        return "fast convergence" if trace_length < 80 else "steady convergence"
    if final_loss < initial_loss * 0.2:
        return "slow convergence"
    return "stalled"
