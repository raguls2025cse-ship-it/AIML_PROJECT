from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "vehicle_detector" / "weights" / "best.pt"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained model found at {MODEL_PATH}. Train the detector first.")
    model = YOLO(str(MODEL_PATH))
    val_dir = ROOT / "data" / "processed" / "yolov8" / "val"
    metrics = model.val(data=str(ROOT / "backend" / "training" / "vehicle_detector.yaml"), split="val")
    print(metrics)


if __name__ == "__main__":
    main()
