from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "vehicle_detector" / "weights" / "best.pt"
FALLBACK_MODEL = "yolov8n.pt"


def get_detector_model():
    if MODEL_PATH.exists():
        return YOLO(str(MODEL_PATH))
    return YOLO(FALLBACK_MODEL)


def run_inference(image_path):
    model = get_detector_model()
    results = model(image_path, conf=0.25, imgsz=640, verbose=False)
    return results[0]
