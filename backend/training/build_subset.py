import json
import random
from collections import Counter
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("subset_config.json")


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_bdd_items(labels_path: Path):
    if not labels_path.exists():
        raise FileNotFoundError(f"BDD labels not found at {labels_path}")
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    items = []
    for item in payload:
        name = item.get("name")
        if not name:
            continue
        labels = item.get("labels", [])
        if not labels:
            continue
        items.append({"name": name, "labels": labels, "width": item.get("width", 1280), "height": item.get("height", 720)})
    return items


def choose_subset(items, target_count, allowed_classes, min_per_class):
    counts = Counter()
    selected = []
    for item in items:
        categories = [lab.get("category") for lab in item["labels"] if lab.get("category")]
        for cat in categories:
            if cat in allowed_classes:
                counts[cat] += 1

    class_candidates = {cls: [] for cls in allowed_classes}
    for item in items:
        categories = {lab.get("category") for lab in item["labels"] if lab.get("category")}
        if not categories:
            continue
        if categories & set(allowed_classes):
            class_candidates[min(categories & set(allowed_classes), key=lambda c: counts[c])].append(item)

    # Collect balanced candidates based on class distribution
    for cls in allowed_classes:
        bucket = []
        for item in items:
            categories = {lab.get("category") for lab in item["labels"] if lab.get("category")}
            if cls in categories:
                bucket.append(item)
        random.shuffle(bucket)
        bucket = bucket[: max(min_per_class, len(bucket))]
        selected.extend(bucket)

    seen = set()
    balanced = []
    for item in selected:
        if item["name"] in seen:
            continue
        seen.add(item["name"])
        balanced.append(item)

    if len(balanced) >= target_count:
        random.shuffle(balanced)
        return balanced[:target_count]

    # fallback: include all unique items in order
    remaining = [item for item in items if item["name"] not in seen]
    random.shuffle(remaining)
    balanced.extend(remaining)
    random.shuffle(balanced)
    return balanced[:target_count]


def write_summary(output_dir: Path, train_items, val_items, test_items):
    stats = {
        "train": len(train_items),
        "val": len(val_items),
        "test": len(test_items),
        "total": len(train_items) + len(val_items) + len(test_items),
        "split": {"train": str(output_dir / "train"), "val": str(output_dir / "val"), "test": str(output_dir / "test")},
    }
    summary_path = output_dir / "subset_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main():
    cfg = load_config()
    bdd_labels = ROOT / cfg["bdd_root"] / "labels" / "det_train.json"
    bdd_images = ROOT / cfg["bdd_root"] / "images" / "100k" / "train"
    train_root = ROOT / cfg["dataset_root"] / "processed" / "subset" / "train"
    val_root = ROOT / cfg["dataset_root"] / "processed" / "subset" / "val"
    test_root = ROOT / cfg["dataset_root"] / "processed" / "subset" / "test"

    if not bdd_labels.exists():
        raise FileNotFoundError(f"BDD labels are missing. Add the downloaded dataset under {bdd_labels}.")

    items = iter_bdd_items(bdd_labels)
    allowed = set(cfg["allowed_classes"])

    all_valid = [item for item in items if any(lab.get("category") in allowed for lab in item["labels"])]
    random.shuffle(all_valid)

    train_target = int(cfg["train_size"])
    val_target = int(cfg["val_size"])
    test_target = int(cfg["test_size"])

    train_items = all_valid[:train_target]
    remaining = all_valid[train_target:]
    val_items = remaining[:val_target]
    test_items = remaining[val_target:val_target + test_target]

    for split_name, split_root, split_items in [("train", train_root, train_items), ("val", val_root, val_items), ("test", test_root, test_items)]:
        image_dir = split_root / "images"
        label_dir = split_root / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for item in split_items:
            src = bdd_images / item["name"]
            if not src.exists():
                continue
            dest = image_dir / item["name"]
            dest.write_bytes(src.read_bytes())
            label_lines = []
            for lab in item["labels"]:
                cat = lab.get("category")
                if cat not in allowed or "box2d" not in lab:
                    continue
                box = lab["box2d"]
                x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
                w = float(item.get("width", 1280))
                h = float(item.get("height", 720))
                if w <= 0 or h <= 0:
                    continue
                label_lines.append(f"{cat} {((x1 + x2) / 2 / w):.6f} {((y1 + y2) / 2 / h):.6f} {((x2 - x1) / w):.6f} {((y2 - y1) / h):.6f}")
            (label_dir / (src.stem + ".txt")).write_text("\n".join(label_lines), encoding="utf-8")

    stats = write_summary(ROOT / cfg["dataset_root"] / "processed" / "subset", train_items, val_items, test_items)
    print(json.dumps({"message": "subset created", **stats}, indent=2))


if __name__ == "__main__":
    main()
