from typing import Dict

from fastapi import APIRouter

from models.rnn import MODEL
from schemas import RnnPredictRequest


router = APIRouter(prefix="/api/rnn", tags=["rnn-next-word"])


@router.post("/predict")
def predict_next_word(request: RnnPredictRequest) -> Dict[str, object]:
    return MODEL.predict_next(request.text, request.top_k)
