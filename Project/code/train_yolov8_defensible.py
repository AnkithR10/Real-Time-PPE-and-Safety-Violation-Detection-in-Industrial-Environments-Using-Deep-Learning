
from pathlib import Path
import csv
import json
import time

import torch
from ultralytics import YOLO


# =====================================================
# CONFIGURATION
# =====================================================

BASE = Path(r"C:\Users\Shreevatsa\Desktop\dl")
DATA = BASE / "code" / "ppe_data.yaml"
OUT = BASE / "results" / "yolov8_baseline_5epochs"

EPOCHS = 5
IMAGE_SIZE = 512
BATCH_SIZE = 4       # Same batch size
WORKERS = 2
DEVICE = 0
SEED = 42


def main():
    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable. Stopping rather than using CPU."
        )

    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("YOLOv8n BASELINE EXPERIMENT")
    print("=" * 60)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Epochs:", EPOCHS)
    print("Training fraction: 100%")
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)

    # Load pretrained YOLOv8 Nano weights.
    model = YOLO("yolov8n.pt")

    start = time.time()

    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,

        project=str(OUT),
        name="yolov8n_baseline",
        exist_ok=False,

        pretrained=True,
        fraction=1.0,
        amp=True,
        cache=False,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=1.0,

        patience=EPOCHS + 1,
        cos_lr=True,
        close_mosaic=0,

        seed=SEED,
        deterministic=True,

        plots=True,
        save=True,
        save_period=1,
        val=True,
        verbose=True,
    )

    elapsed = round((time.time() - start) / 60, 2)

    run_dir = OUT / "yolov8n_baseline"
    best_path = run_dir / "weights" / "best.pt"

    if not best_path.is_file():
        raise FileNotFoundError(f"Best checkpoint missing: {best_path}")

    # Record the experimental configuration.
    config = {
        "model": "YOLOv8n",
        "initialization": "COCO pretrained weights",
        "epochs_requested": EPOCHS,
        "training_fraction": 1.0,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "seed": SEED,
        "training_minutes": elapsed,
        "best_checkpoint": str(best_path),
    }

    with (run_dir / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=4)

    print("\nEvaluating best checkpoint on held-out test set...")

    best_model = YOLO(str(best_path))

    metrics = best_model.val(
        data=str(DATA),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        plots=True,
        project=str(OUT),
        name="yolov8n_baseline_test",
        exist_ok=False,
        verbose=True,
    )

    box = metrics.box

    result = {
        "model": "YOLOv8n",
        "status": "completed",
        "epochs": EPOCHS,
        "training_fraction": 1.0,
        "training_minutes": elapsed,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "best_checkpoint": str(best_path),
    }

    report = OUT / "baseline_test_metrics.csv"

    with report.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    print("\n" + "=" * 60)
    print("BASELINE COMPLETE")
    print("=" * 60)
    print(f"Training time: {elapsed:.2f} minutes")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"mAP@50: {result['mAP50']:.4f}")
    print(f"mAP@50-95: {result['mAP50_95']:.4f}")
    print("Metrics CSV:", report)
    print("Output folder:", OUT)


if __name__ == "__main__":
    main()
