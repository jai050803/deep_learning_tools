from math import exp, tanh
from typing import Tuple

from fastapi import HTTPException


def sigmoid(value: float) -> float:
    if value < -60:
        return 0.0
    if value > 60:
        return 1.0
    return 1 / (1 + exp(-value))


def activation_value(function_name: str, value: float) -> Tuple[float, float]:
    if function_name == "relu":
        return max(0.0, value), 1.0 if value > 0 else 0.0
    if function_name == "sigmoid":
        y_value = sigmoid(value)
        return y_value, y_value * (1 - y_value)
    if function_name == "tanh":
        y_value = tanh(value)
        return y_value, 1 - y_value * y_value
    raise HTTPException(status_code=400, detail="function must be relu, sigmoid, or tanh")


def gradient_state(function_name: str, gradient: float) -> str:
    if function_name == "relu" and gradient == 0:
        return "dead"
    if abs(gradient) < 0.05:
        return "vanishing"
    return "healthy"
