import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data"
ALLOWED = {
    "person": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "traffic light": 8,
    "traffic sign": 9,
}


def build_yolo_split(split_name: str, image_dir: Path, label_dir: Path):
    split_path = DATASET_ROOT / "processed" / "yolov8" / split_name
    (split_path / "images").mkdir(parents=True, exist_ok=True)
    (split_path / "labels").mkdir(parents=True, exist_ok=True)

    for img in sorted(image_dir.glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        label_file = label_dir / (img.stem + ".txt")
        target_image = split_path / "images" / img.name
        target_label = split_path / "labels" / label_file.name
        if img.exists():
            target_image.write_bytes(img.read_bytes())
        if label_file.exists():
            target_label.write_text(label_file.read_text(encoding="utf-8"), encoding="utf-8")


def main():
    subset_root = DATASET_ROOT / "processed" / "subset"
    for split_name in ["train", "val", "test"]:
        image_dir = subset_root / split_name / "images"
        label_dir = subset_root / split_name / "labels"
        if not image_dir.exists():
            print(f"Skipping {split_name}: no subset data created yet.")
            continue
        build_yolo_split(split_name, image_dir, label_dir)
        print(f"Prepared YOLO split: {split_name} ({len(list(image_dir.glob('*')))} images)")


if __name__ == "__main__":
    main()
