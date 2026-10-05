import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "vehicle_detector"
DATASET_DIR = ROOT / "data" / "processed" / "yolov8"


def main():
    config_path = Path(__file__).with_name("subset_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    train_split = DATASET_DIR / "train"
    val_split = DATASET_DIR / "val"
    if not train_split.exists() or not val_split.exists():
        raise FileNotFoundError("Run build_subset.py and prepare_detection_dataset.py before training.")

    model = YOLO("yolov8n.pt")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = model.train(
        data=str(ROOT / "backend" / "training" / "vehicle_detector.yaml"),
        epochs=int(config.get("epochs", 15)),
        imgsz=int(config.get("image_size", 640)),
        batch=int(config.get("batch_size", 8)),
        project=str(MODEL_DIR.parent),
        name="vehicle_detector",
        exist_ok=True,
        pretrained=True,
        device="0" if __import__('torch').cuda.is_available() else "cpu",
    )
    print(results)


if __name__ == "__main__":
    main()
