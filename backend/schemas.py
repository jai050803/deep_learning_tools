from pydantic import BaseModel, Field


class Weights(BaseModel):
    w1: float = Field(default=0.2, ge=-5.0, le=5.0)
    w2: float = Field(default=-0.35, ge=-5.0, le=5.0)
    bias: float = Field(default=0.1, ge=-5.0, le=5.0)


class TrainRequest(BaseModel):
    dataset: str = Field(default="AND")
    model_type: str = Field(default="perceptron")
    learning_rate: float = Field(default=0.2, ge=0.001, le=2.0)
    epochs: int = Field(default=24, ge=0, le=5000)
    train: bool = Field(default=True)
    weights: Weights = Field(default_factory=Weights)
    grid_steps: int = Field(default=45, ge=20, le=80)


class ActivationRequest(BaseModel):
    function: str = Field(default="relu")
    x_min: float = Field(default=-6.0, ge=-20.0, le=0.0)
    x_max: float = Field(default=6.0, ge=0.0, le=20.0)
    points: int = Field(default=161, ge=40, le=500)


class GradientDescentRequest(BaseModel):
    learning_rate: float = Field(default=0.08, ge=0.001, le=1.2)
    epochs: int = Field(default=48, ge=1, le=400)
    initial_weight: float = Field(default=-4.0, ge=-10.0, le=10.0)
    target_weight: float = Field(default=3.0, ge=-10.0, le=10.0)


class RnnPredictRequest(BaseModel):
    text: str = Field(default="Hello how are", min_length=1, max_length=240)
    top_k: int = Field(default=3, ge=1, le=5)


class ExplainModelRequest(BaseModel):
    dataset: str = Field(default="XOR")
    model_type: str = Field(default="mlp")
    learning_rate: float = Field(default=0.2, ge=0.001, le=2.0)
    epochs: int = Field(default=120, ge=1, le=5000)
    hidden_neurons: int = Field(default=4, ge=1, le=16)
    train_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    validation_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    final_loss: float | None = Field(default=None, ge=0.0)
    loss_volatility: float | None = Field(default=None, ge=0.0)


class WhatIfRequest(BaseModel):
    dataset: str = Field(default="XOR")
    hidden_neurons: int = Field(default=4, ge=1, le=16)
    learning_rate: float = Field(default=0.2, ge=0.001, le=2.0)
    epochs: int = Field(default=120, ge=1, le=1000)
    noise: float = Field(default=0.08, ge=0.0, le=0.4)
