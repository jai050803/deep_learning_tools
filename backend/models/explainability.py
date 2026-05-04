from math import exp, sin
from typing import Dict, List

from fastapi import HTTPException

from models.datasets import DATASETS
from schemas import ExplainModelRequest, WhatIfRequest


DATASET_COMPLEXITY = {
    "AND": 1.0,
    "OR": 1.0,
    "XOR": 2.4,
}


def normalize_dataset(name: str) -> str:
    dataset = name.strip().upper()
    if dataset not in DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset '{name}'")
    return dataset


def _learning_rate_instability(learning_rate: float) -> float:
    too_high = max(0.0, learning_rate - 0.72) / 1.28
    too_low = max(0.0, 0.025 - learning_rate) / 0.025
    return min(1.0, too_high * 1.15 + too_low * 0.35)


def simulate_what_if(request: WhatIfRequest) -> Dict[str, object]:
    dataset = normalize_dataset(request.dataset)
    complexity = DATASET_COMPLEXITY[dataset]
    capacity = min(1.15, request.hidden_neurons / (complexity + 0.8))
    progress = 1 - exp(-request.epochs / (52 + complexity * 34))
    instability = _learning_rate_instability(request.learning_rate)
    underfit = max(0.0, 1.0 - capacity)
    overfit_pressure = max(0.0, request.hidden_neurons - (complexity + 3.0)) / 12.0
    overfit_pressure *= min(1.0, request.epochs / 420) * (0.45 + request.noise)

    train_accuracy = 0.48 + 0.49 * min(1.0, capacity) * progress
    train_accuracy -= 0.18 * instability + 0.18 * underfit
    train_accuracy = max(0.25, min(0.995, train_accuracy))

    validation_accuracy = train_accuracy
    validation_accuracy -= 0.24 * overfit_pressure
    validation_accuracy -= 0.12 * instability
    validation_accuracy -= 0.08 * request.noise
    validation_accuracy = max(0.2, min(0.98, validation_accuracy))

    loss_trace = []
    start_loss = 1.1 + complexity * 0.22 + request.noise * 0.7
    floor = max(0.025, 0.65 - train_accuracy * 0.58)
    samples = 32
    for step in range(samples):
        ratio = step / (samples - 1)
        decay = exp(-ratio * (2.2 + progress * 3.4))
        oscillation = instability * (0.12 + ratio * 0.2) * sin(step * 1.7)
        overfit_bump = overfit_pressure * max(0.0, ratio - 0.62) * 0.34
        loss = max(0.01, floor + (start_loss - floor) * decay + oscillation + overfit_bump)
        loss_trace.append({"epoch": round(1 + ratio * (request.epochs - 1)), "loss": round(loss, 5)})

    gap = max(0.0, train_accuracy - validation_accuracy)
    if request.learning_rate > 0.72 or instability > 0.28:
        behavior = "unstable"
    elif gap > 0.14:
        behavior = "overfitting"
    elif train_accuracy < 0.72 and validation_accuracy < 0.72:
        behavior = "underfitting"
    else:
        behavior = "healthy"

    return {
        "dataset": dataset,
        "hidden_neurons": request.hidden_neurons,
        "learning_rate": request.learning_rate,
        "epochs": request.epochs,
        "noise": request.noise,
        "train_accuracy": round(train_accuracy, 4),
        "validation_accuracy": round(validation_accuracy, 4),
        "generalization_gap": round(gap, 4),
        "loss_volatility": round(instability, 4),
        "overfit_pressure": round(overfit_pressure, 4),
        "loss_trace": loss_trace,
        "behavior": behavior,
    }


def explain_model(request: ExplainModelRequest) -> Dict[str, object]:
    dataset = normalize_dataset(request.dataset)
    simulated = simulate_what_if(
        WhatIfRequest(
            dataset=dataset,
            hidden_neurons=request.hidden_neurons,
            learning_rate=request.learning_rate,
            epochs=min(request.epochs, 1000),
            noise=0.08,
        )
    )

    train_accuracy = request.train_accuracy
    if train_accuracy is None:
        train_accuracy = simulated["train_accuracy"]
    validation_accuracy = request.validation_accuracy
    if validation_accuracy is None:
        validation_accuracy = simulated["validation_accuracy"]
    loss_volatility = request.loss_volatility
    if loss_volatility is None:
        loss_volatility = simulated["loss_volatility"]

    gap = max(0.0, train_accuracy - validation_accuracy)
    insights: List[Dict[str, str]] = []

    if request.learning_rate > 0.72 or loss_volatility > 0.42:
        insights.append(
            {
                "type": "warning",
                "title": "Learning rate zyada hai",
                "message": "Loss curve oscillate kar sakti hai, isliye model stable minimum ke around jump kar raha hai.",
            }
        )
    elif request.learning_rate < 0.025:
        insights.append(
            {
                "type": "info",
                "title": "Learning rate bahut low hai",
                "message": "Training safe hai, but convergence slow hoga. Epochs badhane padenge.",
            }
        )

    if gap >= 0.14:
        insights.append(
            {
                "type": "warning",
                "title": "Model overfit ho raha hai",
                "message": "Training accuracy high hai but validation accuracy noticeably lower hai. Generalization weak hai.",
            }
        )

    if train_accuracy >= 0.92 and validation_accuracy < 0.82:
        insights.append(
            {
                "type": "warning",
                "title": "Accuracy high hai, par trust mat karo",
                "message": "Model training examples yaad kar raha hai. New data par performance drop ho sakti hai.",
            }
        )

    if train_accuracy < 0.72 and validation_accuracy < 0.72:
        insights.append(
            {
                "type": "info",
                "title": "Underfitting signal",
                "message": "Model capacity ya epochs kam lag rahe hain. Neurons ya epochs thoda increase karke dekho.",
            }
        )

    if dataset == "XOR" and request.model_type.strip().lower() == "perceptron":
        insights.append(
            {
                "type": "warning",
                "title": "XOR linear boundary se solve nahi hota",
                "message": "Perceptron ek straight line banata hai. XOR ke liye hidden layer wala MLP better choice hai.",
            }
        )

    if not insights:
        insights.append(
            {
                "type": "success",
                "title": "Model healthy lag raha hai",
                "message": "Training aur validation close hain, loss stable hai, aur learning rate controlled range mein hai.",
            }
        )

    recommendations = []
    if gap >= 0.14:
        recommendations.extend(["Hidden neurons kam karo", "Epochs reduce karo", "Validation accuracy ko primary metric banao"])
    if request.learning_rate > 0.72 or loss_volatility > 0.42:
        recommendations.append("Learning rate 0.05 se 0.3 ke beech try karo")
    if train_accuracy < 0.72:
        recommendations.extend(["Epochs badhao", "MLP use karo", "Neurons gradually increase karo"])
    if not recommendations:
        recommendations.append("Current settings ko baseline save karke nearby settings compare karo")

    score = max(0.0, min(1.0, validation_accuracy - gap * 0.35 - loss_volatility * 0.18))
    return {
        "dataset": dataset,
        "model_type": request.model_type,
        "train_accuracy": round(train_accuracy, 4),
        "validation_accuracy": round(validation_accuracy, 4),
        "generalization_gap": round(gap, 4),
        "loss_volatility": round(float(loss_volatility), 4),
        "health_score": round(score, 4),
        "insights": insights,
        "recommendations": recommendations[:5],
    }
