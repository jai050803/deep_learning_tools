import base64
from pathlib import Path
from typing import Dict, List

from fastapi import HTTPException

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


def encode_image(image) -> str:
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode processed image")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


def decode_upload(contents: bytes):
    if cv2 is None or np is None:
        raise HTTPException(
            status_code=503,
            detail="OpenCV is not installed. Run pip install -r requirements.txt first.",
        )
    array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Upload a valid image file")
    return image


def classify_image(image) -> List[Dict[str, object]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = float(gray.mean())
    saturation = float(hsv[:, :, 1].mean())
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float((edges > 0).mean())

    labels = []
    if detect_faces(image):
        labels.append({"label": "person / face", "confidence": 0.91})
    if edge_density > 0.13:
        labels.append(
            {
                "label": "structured object scene",
                "confidence": round(min(0.88, edge_density * 5), 2),
            }
        )
    if saturation > 95:
        labels.append({"label": "color-rich image", "confidence": round(min(0.86, saturation / 150), 2)})
    if brightness < 85:
        labels.append({"label": "low-light scene", "confidence": 0.78})
    elif brightness > 175:
        labels.append({"label": "bright scene", "confidence": 0.76})

    if not labels:
        labels.append({"label": "general image", "confidence": 0.68})
    return labels[:3]


def detect_faces(image) -> List[Dict[str, object]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(36, 36))
    boxes = []
    for index, (x, y, w, h) in enumerate(faces):
        boxes.append(
            {
                "label": "face",
                "confidence": 0.88,
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
                "id": index + 1,
            }
        )
    return boxes


def detect_salient_regions(image) -> List[Dict[str, object]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 70, 170)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < image_area * 0.015 or area > image_area * 0.72:
            continue
        regions.append(
            {
                "label": "salient region",
                "confidence": round(min(0.72, 0.35 + area / image_area), 2),
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
            }
        )
    regions.sort(key=lambda item: item["width"] * item["height"], reverse=True)
    return regions[:6]


def draw_boxes(image, boxes: List[Dict[str, object]]):
    output = image.copy()
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        color = (38, 132, 255) if box["label"] == "face" else (38, 184, 128)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 3)
        label = f"{box['label']} {int(box['confidence'] * 100)}%"
        cv2.rectangle(output, (x, max(0, y - 26)), (x + max(120, len(label) * 9), y), color, -1)
        cv2.putText(
            output,
            label,
            (x + 6, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
    return output
