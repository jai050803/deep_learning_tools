from typing import Dict, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from models.vision import (
    classify_image,
    decode_upload,
    detect_faces,
    detect_salient_regions,
    draw_boxes,
    encode_image,
)


router = APIRouter(prefix="/api/vision", tags=["opencv-vision"])


@router.post("/analyze")
async def analyze_image(
    task: str = Form(default="object_detection"),
    image: UploadFile = File(...),
) -> Dict[str, object]:
    contents = await image.read()
    decoded = decode_upload(contents)
    height, width = decoded.shape[:2]
    task_name = task.strip().lower()
    if task_name not in {"image_classification", "object_detection", "face_detection"}:
        raise HTTPException(
            status_code=400,
            detail="task must be image_classification, object_detection, or face_detection",
        )

    predictions = classify_image(decoded)
    boxes: List[Dict[str, object]] = []
    if task_name in {"object_detection", "face_detection"}:
        boxes = detect_faces(decoded)
        if task_name == "object_detection":
            boxes = boxes + detect_salient_regions(decoded)

    annotated = draw_boxes(decoded, boxes) if boxes else decoded
    return {
        "task": task_name,
        "width": width,
        "height": height,
        "model": "OpenCV Haar Cascade + lightweight visual analyzer",
        "predictions": predictions,
        "boxes": boxes,
        "annotated_image": encode_image(annotated),
        "note": "YOLO/ResNet weights can be plugged into backend/models later; this endpoint is Render-safe and uses OpenCV's bundled pretrained face detector now.",
    }
