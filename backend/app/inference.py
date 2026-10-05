import os
from pathlib import Path

import cv2
from ultralytics import YOLO

from .visibility import visibility_score, visibility_status
from .risk import assess_risk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "vehicle_detector" / "weights" / "best.pt"
FALLBACK_MODEL = "yolov8n.pt"
ALLOWED = {"person", "car", "truck", "bus", "motorcycle", "bicycle", "train", "traffic light", "traffic sign"}


def _load_detector_model():
    if MODEL_PATH.exists():
        return YOLO(str(MODEL_PATH))
    return YOLO(os.getenv("YOLO_MODEL", FALLBACK_MODEL))


model = _load_detector_model()


def detect_frame(frame):
    h, w = frame.shape[:2]
    result = model.predict(frame, verbose=False, conf=0.35, imgsz=640)[0]
    detections = []
    if result.boxes is not None:
        for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            label = model.names[int(cls)]
            if label not in ALLOWED:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append({"label": label, "confidence": round(float(conf), 3), "x1": x1, "y1": y1, "x2": x2, "y2": y2, "risk": "LOW"})
    score, status = visibility_score(frame)
    if not isinstance(score, (int, float)):
        score = 0.0
    risk, warning = assess_risk(detections, w, h, status)
    annotated = frame.copy()
    for d in detections:
        color = (0, 0, 255) if d["risk"] == "HIGH" else (0, 165, 255) if d["risk"] == "CAUTION" else (0, 200, 0)
        x1, y1, x2, y2 = map(int, [d["x1"], d["y1"], d["x2"], d["y2"]])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f'{d["label"]} {d["confidence"]:.2f}', (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(annotated, f"Visibility: {status} ({score:.0f})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"Risk: {risk}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if "HIGH" in risk else (0, 165, 255), 2)
    return annotated, float(score), status, risk, warning, detections
