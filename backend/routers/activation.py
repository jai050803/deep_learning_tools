from typing import Dict

from fastapi import APIRouter, HTTPException

from models.activation import activation_value, gradient_state
from schemas import ActivationRequest


router = APIRouter(prefix="/api/activation", tags=["activation-functions"])


@router.post("/curve")
def activation_curve(request: ActivationRequest) -> Dict[str, object]:
    function_name = request.function.strip().lower()
    if request.x_min >= request.x_max:
        raise HTTPException(status_code=400, detail="x_min must be less than x_max")

    step = (request.x_max - request.x_min) / (request.points - 1)
    samples = []
    vanishing_count = 0
    for index in range(request.points):
        x_value = request.x_min + step * index
        y_value, gradient = activation_value(function_name, x_value)
        state = gradient_state(function_name, gradient)
        if state in {"dead", "vanishing"}:
            vanishing_count += 1
        samples.append(
            {
                "x": round(x_value, 5),
                "y": round(y_value, 5),
                "gradient": round(gradient, 5),
                "state": state,
            }
        )

    return {
        "function": function_name,
        "samples": samples,
        "vanishing_ratio": round(vanishing_count / len(samples), 4),
        "max_gradient": max(point["gradient"] for point in samples),
        "concept": "Gradients near zero make earlier layers learn slowly. Sigmoid and tanh saturate at the edges; ReLU can become inactive for negative inputs.",
    }
