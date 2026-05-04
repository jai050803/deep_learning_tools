from typing import Dict

from fastapi import APIRouter, HTTPException

from models.datasets import DATASETS
from models.perceptron_mlp import (
    evaluate_mlp,
    evaluate_perceptron,
    normalize_dataset_name,
    perceptron_boundary,
    prediction_grid,
    train_mlp,
    train_perceptron,
)
from schemas import TrainRequest


router = APIRouter(prefix="/api/perceptron", tags=["perceptron-mlp"])


@router.get("/datasets")
def get_datasets() -> Dict[str, object]:
    return {"datasets": DATASETS}


@router.post("/train")
def train_module(request: TrainRequest) -> Dict[str, object]:
    dataset_name = normalize_dataset_name(request.dataset)
    model_type = request.model_type.strip().lower()
    if model_type not in {"perceptron", "mlp"}:
        raise HTTPException(status_code=400, detail="model_type must be 'perceptron' or 'mlp'")

    dataset = DATASETS[dataset_name]
    perceptron_training = train_perceptron(
        dataset=dataset,
        initial_weights=request.weights,
        learning_rate=request.learning_rate,
        epochs=request.epochs,
        should_train=request.train and model_type == "perceptron",
    )
    weights = perceptron_training["weights"]
    mlp_params = None
    history = perceptron_training["history"]

    if model_type == "mlp":
        mlp_training = train_mlp(dataset, request.learning_rate, max(1, request.epochs))
        mlp_params = mlp_training["params"]
        history = mlp_training["history"]
        evaluated = evaluate_mlp(dataset, mlp_params)
        boundary = None
        concept = "Non-linear MLP: hidden neurons bend the decision regions, so XOR can be solved."
    else:
        evaluated = evaluate_perceptron(dataset, weights)
        boundary = perceptron_boundary(weights)
        concept = "Linear perceptron: one straight decision boundary works for AND/OR, but not XOR."

    return {
        "dataset": dataset_name,
        "model_type": model_type,
        "learning_rate": request.learning_rate,
        "epochs": request.epochs,
        "accuracy": evaluated["accuracy"],
        "points": evaluated["points"],
        "weights": weights,
        "mlp_params": mlp_params,
        "boundary": boundary,
        "grid": prediction_grid(model_type, request.grid_steps, weights, mlp_params),
        "history": history,
        "concept": concept,
    }
