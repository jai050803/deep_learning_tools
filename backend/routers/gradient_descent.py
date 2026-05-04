from typing import Dict

from fastapi import APIRouter

from models.gradient_descent import classify_convergence, gd_gradient, gd_loss
from schemas import GradientDescentRequest


router = APIRouter(prefix="/api/gradient-descent", tags=["gradient-descent"])


@router.post("/simulate")
def gradient_descent_simulation(request: GradientDescentRequest) -> Dict[str, object]:
    weight = request.initial_weight
    trace = []
    diverged = False

    for epoch in range(request.epochs + 1):
        loss = gd_loss(weight, request.target_weight)
        grad = gd_gradient(weight, request.target_weight)
        trace.append(
            {
                "epoch": epoch,
                "weight": round(weight, 5),
                "loss": round(loss, 5),
                "gradient": round(grad, 5),
            }
        )
        if epoch == request.epochs:
            break
        weight = weight - request.learning_rate * grad
        if abs(weight) > 1_000_000:
            diverged = True
            break

    final_loss = trace[-1]["loss"]
    convergence = classify_convergence(diverged, trace[0]["loss"], final_loss, len(trace))

    return {
        "learning_rate": request.learning_rate,
        "epochs": request.epochs,
        "trace": trace,
        "final_loss": final_loss,
        "convergence": convergence,
        "concept": "Small learning rates crawl, balanced rates converge cleanly, and large rates overshoot or diverge.",
    }
