
from pathlib import Path
from collections import Counter
import math

# Dataset configuration
ROOT = Path(r"C:\Users\Shreevatsa\Desktop\dl\archive")
SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUM_CLASSES = 14

CLASS_NAMES = [
    "Fall-Detected",
    "Gloves",
    "Goggles",
    "Hardhat",
    "Ladder",
    "Mask",
    "NO-Gloves",
    "NO-Goggles",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest"
]


def check_split(split):
    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    print("\n" + "=" * 65)
    print(f"CHECKING {split.upper()} DATASET")
    print("=" * 65)

    if not image_dir.is_dir():
        print(f"ERROR: Image directory not found: {image_dir}")
        return None

    if not label_dir.is_dir():
        print(f"ERROR: Label directory not found: {label_dir}")
        return None

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))

    image_map = {p.stem: p for p in images}
    label_map = {p.stem: p for p in labels}

    missing_labels = sorted(set(image_map) - set(label_map))
    orphan_labels = sorted(set(label_map) - set(image_map))

    empty_labels = []
    invalid_rows = []
    class_counts = Counter()
    total_boxes = 0
    readable_labels = 0

    for name, label_path in label_map.items():
        try:
            lines = label_path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
        except (OSError, UnicodeError) as exc:
            invalid_rows.append(
                (name, f"Cannot read label file: {exc}")
            )
            continue

        rows = [line.strip() for line in lines if line.strip()]

        if not rows:
            empty_labels.append(name)
            readable_labels += 1
            continue

        readable_labels += 1

        for line_number, row in enumerate(rows, start=1):
            parts = row.split()

            if len(parts) != 5:
                invalid_rows.append(
                    (name, line_number, "Expected 5 values")
                )
                continue

            try:
                values = [float(value) for value in parts]
            except ValueError:
                invalid_rows.append(
                    (name, line_number, "Non-numeric values")
                )
                continue

            if not all(math.isfinite(v) for v in values):
                invalid_rows.append(
                    (name, line_number, "NaN or infinite value")
                )
                continue

            class_value, x, y, w, h = values

            if not class_value.is_integer():
                invalid_rows.append(
                    (name, line_number, "Class ID is not an integer")
                )
                continue

            class_id = int(class_value)

            if not 0 <= class_id < NUM_CLASSES:
                invalid_rows.append(
                    (name, line_number, "Class ID outside 0-13")
                )
                continue

            if not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < w <= 1
                and 0 < h <= 1
            ):
                invalid_rows.append(
                    (name, line_number, "Coordinates or dimensions out of range")
                )
                continue

            # Check whether the bounding box stays inside the image
            if (
                x - w / 2 < -1e-6
                or x + w / 2 > 1 + 1e-6
                or y - h / 2 < -1e-6
                or y + h / 2 > 1 + 1e-6
            ):
                invalid_rows.append(
                    (name, line_number, "Bounding box extends outside image")
                )
                continue

            class_counts[class_id] += 1
            total_boxes += 1

    print(f"Images:                    {len(images)}")
    print(f"Label files:               {len(labels)}")
    print(f"Readable label files:      {readable_labels}")
    print(f"Images without labels:     {len(missing_labels)}")
    print(f"Labels without images:     {len(orphan_labels)}")
    print(f"Empty label files:         {len(empty_labels)}")
    print(f"Valid bounding boxes:      {total_boxes}")
    print(f"Invalid label rows:        {len(invalid_rows)}")

    print("\nObject counts by class:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_id:2} | {class_name:<18} | "
              f"{class_counts[class_id]}")

    if missing_labels:
        print("\nFirst 10 images without labels:")
        for name in missing_labels[:10]:
            print(" ", name)

    if orphan_labels:
        print("\nFirst 10 labels without images:")
        for name in orphan_labels[:10]:
            print(" ", name)

    if empty_labels:
        print("\nFirst 10 empty label files:")
        for name in empty_labels[:10]:
            print(" ", name)

    if invalid_rows:
        print("\nFirst 15 invalid annotation rows:")
        for item in invalid_rows[:15]:
            print(" ", item)

    return {
        "split": split,
        "images": len(images),
        "labels": len(labels),
        "missing": len(missing_labels),
        "orphan": len(orphan_labels),
        "empty": len(empty_labels),
        "boxes": total_boxes,
        "invalid": len(invalid_rows),
        "classes": class_counts
    }


if __name__ == "__main__":
    results = []

    for split in SPLITS:
        result = check_split(split)
        if result is not None:
            results.append(result)

    print("\n" + "=" * 85)
    print("FINAL DATASET SUMMARY")
    print("=" * 85)

    print(
        f"{'Split':<10}{'Images':>10}{'Labels':>10}"
        f"{'Missing':>10}{'Orphans':>10}{'Empty':>10}"
        f"{'Boxes':>12}{'Bad rows':>12}"
    )

    for r in results:
        print(
            f"{r['split']:<10}{r['images']:>10}"
            f"{r['labels']:>10}{r['missing']:>10}"
            f"{r['orphan']:>10}{r['empty']:>10}"
            f"{r['boxes']:>12}{r['invalid']:>12}"
        )

    print("\nDataset inspection completed.")
    print("No dataset files were modified.")
