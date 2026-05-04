from typing import Dict

from fastapi import APIRouter

from models.explainability import explain_model, simulate_what_if
from schemas import ExplainModelRequest, WhatIfRequest


router = APIRouter(prefix="/api/insights", tags=["model-insights"])


@router.post("/explain")
def explain(request: ExplainModelRequest) -> Dict[str, object]:
    return explain_model(request)


@router.post("/what-if")
def what_if(request: WhatIfRequest) -> Dict[str, object]:
    return simulate_what_if(request)
