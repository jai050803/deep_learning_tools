from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import activation, explainability, gradient_descent, perceptron, rnn, vision


app = FastAPI(
    title="Deep Learning Toolbox API",
    description="Interactive model logic for perceptron, MLP, activation, gradient descent, vision, and RNN modules.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(perceptron.router)
app.include_router(activation.router)
app.include_router(gradient_descent.router)
app.include_router(explainability.router)
app.include_router(vision.router)
app.include_router(rnn.router)


@app.get("/")
def read_root() -> Dict[str, object]:
    return {
        "message": "Deep Learning Toolbox API",
        "modules": [
            "perceptron-mlp",
            "activation-functions",
            "gradient-descent",
            "model-insights",
            "what-if-playground",
            "opencv-vision",
            "rnn-next-word",
        ],
        "docs": "/docs",
    }


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}
