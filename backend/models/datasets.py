from typing import Dict, List


DATASETS: Dict[str, List[Dict[str, float]]] = {
    "AND": [
        {"x1": 0.0, "x2": 0.0, "target": 0.0},
        {"x1": 0.0, "x2": 1.0, "target": 0.0},
        {"x1": 1.0, "x2": 0.0, "target": 0.0},
        {"x1": 1.0, "x2": 1.0, "target": 1.0},
    ],
    "OR": [
        {"x1": 0.0, "x2": 0.0, "target": 0.0},
        {"x1": 0.0, "x2": 1.0, "target": 1.0},
        {"x1": 1.0, "x2": 0.0, "target": 1.0},
        {"x1": 1.0, "x2": 1.0, "target": 1.0},
    ],
    "XOR": [
        {"x1": 0.0, "x2": 0.0, "target": 0.0},
        {"x1": 0.0, "x2": 1.0, "target": 1.0},
        {"x1": 1.0, "x2": 0.0, "target": 1.0},
        {"x1": 1.0, "x2": 1.0, "target": 0.0},
    ],
}
