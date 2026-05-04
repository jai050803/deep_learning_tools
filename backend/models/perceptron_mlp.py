from math import exp
from typing import Dict, List, Optional

from fastapi import HTTPException

from models.datasets import DATASETS
from schemas import Weights


def normalize_dataset_name(name: str) -> str:
    dataset = name.strip().upper()
    if dataset not in DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset '{name}'")
    return dataset


def perceptron_score(x1: float, x2: float, weights: Dict[str, float]) -> float:
    return weights["w1"] * x1 + weights["w2"] * x2 + weights["bias"]


def perceptron_predict(x1: float, x2: float, weights: Dict[str, float]) -> int:
    return 1 if perceptron_score(x1, x2, weights) >= 0 else 0


def accuracy_for_predictions(points: List[Dict[str, float]]) -> float:
    correct = sum(1 for point in points if int(point["target"]) == int(point["prediction"]))
    return round(correct / len(points), 4)


def train_perceptron(
    dataset: List[Dict[str, float]],
    initial_weights: Weights,
    learning_rate: float,
    epochs: int,
    should_train: bool,
) -> Dict[str, object]:
    weights = initial_weights.model_dump()
    history = []
    total_epochs = epochs if should_train else 0

    for epoch in range(total_epochs):
        updates = 0
        for point in dataset:
            prediction = perceptron_predict(point["x1"], point["x2"], weights)
            error = int(point["target"]) - prediction
            if error != 0:
                updates += 1
            weights["w1"] += learning_rate * error * point["x1"]
            weights["w2"] += learning_rate * error * point["x2"]
            weights["bias"] += learning_rate * error

        evaluated = evaluate_perceptron(dataset, weights)
        history.append(
            {
                "epoch": epoch + 1,
                "accuracy": evaluated["accuracy"],
                "updates": updates,
            }
        )

    return {
        "weights": {key: round(value, 5) for key, value in weights.items()},
        "history": history,
    }


def evaluate_perceptron(
    dataset: List[Dict[str, float]], weights: Dict[str, float]
) -> Dict[str, object]:
    points = []
    for point in dataset:
        score = perceptron_score(point["x1"], point["x2"], weights)
        points.append(
            {
                **point,
                "score": round(score, 5),
                "prediction": perceptron_predict(point["x1"], point["x2"], weights),
            }
        )

    return {"points": points, "accuracy": accuracy_for_predictions(points)}


def perceptron_boundary(weights: Dict[str, float]) -> Optional[Dict[str, object]]:
    w1 = weights["w1"]
    w2 = weights["w2"]
    bias = weights["bias"]
    if abs(w2) > 1e-9:
        return {
            "type": "line",
            "points": [
                {"x1": -0.25, "x2": round(-(w1 * -0.25 + bias) / w2, 5)},
                {"x1": 1.25, "x2": round(-(w1 * 1.25 + bias) / w2, 5)},
            ],
        }
    if abs(w1) > 1e-9:
        return {"type": "vertical", "x1": round(-bias / w1, 5)}
    return None


def sigmoid(value: float) -> float:
    if value < -60:
        return 0.0
    if value > 60:
        return 1.0
    return 1 / (1 + exp(-value))


def mlp_forward(x1: float, x2: float, params: Dict[str, object]) -> Dict[str, object]:
    hidden_values = []
    hidden_weights = params["hidden_weights"]
    hidden_biases = params["hidden_biases"]
    output_weights = params["output_weights"]

    for index in range(2):
        z_value = (
            hidden_weights[index][0] * x1
            + hidden_weights[index][1] * x2
            + hidden_biases[index]
        )
        hidden_values.append(sigmoid(z_value))

    output_z = (
        output_weights[0] * hidden_values[0]
        + output_weights[1] * hidden_values[1]
        + params["output_bias"]
    )
    output = sigmoid(output_z)
    return {"hidden": hidden_values, "output": output}


def train_mlp(
    dataset: List[Dict[str, float]], learning_rate: float, epochs: int
) -> Dict[str, object]:
    params: Dict[str, object] = {
        "hidden_weights": [[4.2, -4.1], [-3.9, 4.0]],
        "hidden_biases": [-1.8, -1.9],
        "output_weights": [4.1, 4.0],
        "output_bias": -1.9,
    }
    history = []

    for epoch in range(epochs):
        total_loss = 0.0
        for point in dataset:
            x1 = point["x1"]
            x2 = point["x2"]
            target = point["target"]
            forward = mlp_forward(x1, x2, params)
            hidden = forward["hidden"]
            output = forward["output"]
            error = output - target
            total_loss += error * error

            output_delta = error * output * (1 - output)
            old_output_weights = params["output_weights"][:]

            params["output_weights"][0] -= learning_rate * output_delta * hidden[0]
            params["output_weights"][1] -= learning_rate * output_delta * hidden[1]
            params["output_bias"] -= learning_rate * output_delta

            for index in range(2):
                hidden_delta = (
                    output_delta
                    * old_output_weights[index]
                    * hidden[index]
                    * (1 - hidden[index])
                )
                params["hidden_weights"][index][0] -= learning_rate * hidden_delta * x1
                params["hidden_weights"][index][1] -= learning_rate * hidden_delta * x2
                params["hidden_biases"][index] -= learning_rate * hidden_delta

        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 24) == 0:
            evaluated = evaluate_mlp(dataset, params)
            history.append(
                {
                    "epoch": epoch + 1,
                    "accuracy": evaluated["accuracy"],
                    "loss": round(total_loss / len(dataset), 5),
                }
            )

    return {"params": round_mlp_params(params), "history": history}


def round_mlp_params(params: Dict[str, object]) -> Dict[str, object]:
    return {
        "hidden_weights": [
            [round(value, 5) for value in row] for row in params["hidden_weights"]
        ],
        "hidden_biases": [round(value, 5) for value in params["hidden_biases"]],
        "output_weights": [round(value, 5) for value in params["output_weights"]],
        "output_bias": round(params["output_bias"], 5),
    }


def evaluate_mlp(dataset: List[Dict[str, float]], params: Dict[str, object]) -> Dict[str, object]:
    points = []
    for point in dataset:
        probability = mlp_forward(point["x1"], point["x2"], params)["output"]
        points.append(
            {
                **point,
                "score": round(probability, 5),
                "prediction": 1 if probability >= 0.5 else 0,
            }
        )
    return {"points": points, "accuracy": accuracy_for_predictions(points)}


def prediction_grid(
    model_type: str,
    steps: int,
    weights: Dict[str, float],
    mlp_params: Optional[Dict[str, object]],
) -> List[Dict[str, float]]:
    grid = []
    minimum = -0.25
    maximum = 1.25
    span = maximum - minimum

    for row in range(steps):
        x2 = minimum + span * row / (steps - 1)
        for column in range(steps):
            x1 = minimum + span * column / (steps - 1)
            if model_type == "mlp" and mlp_params is not None:
                score = mlp_forward(x1, x2, mlp_params)["output"]
                prediction = 1 if score >= 0.5 else 0
            else:
                score = perceptron_score(x1, x2, weights)
                prediction = 1 if score >= 0 else 0
            grid.append(
                {
                    "x1": round(x1, 5),
                    "x2": round(x2, 5),
                    "score": round(score, 5),
                    "prediction": prediction,
                }
            )
    return grid
