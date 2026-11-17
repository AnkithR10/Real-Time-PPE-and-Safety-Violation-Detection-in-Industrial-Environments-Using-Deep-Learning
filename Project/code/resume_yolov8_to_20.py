
from pathlib import Path
import torch
from ultralytics import YOLO

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE = Path(r"C:\Users\Shreevatsa\Desktop\dl")

DATA_YAML = BASE / "code" / "ppe_data.yaml"
RESULTS_DIR = BASE / "results"


# ==========================================
# 2. TRAINING CONFIGURATION
# ==========================================

EPOCHS = 20
IMAGE_SIZE = 512
BATCH_SIZE = 4
WORKERS = 2
DEVICE = 0
SEED = 42


# ==========================================
# 3. MAIN FUNCTION
# ==========================================

def main():

    # Check dataset YAML
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    # Check CUDA GPU
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check your PyTorch installation."
        )

    print("=" * 60)
    print("YOLOv8n PPE DETECTION - 20 EPOCHS")
    print("=" * 60)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Dataset:", DATA_YAML)
    print("Epochs:", EPOCHS)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Training data: Full dataset")
    print("Starting from: Pretrained YOLOv8n weights")

    # Load the pretrained YOLOv8n model.
    # This starts a fresh training run, not from your old checkpoint.
    model = YOLO("yolov8n.pt")

    # Train
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,

        project=str(RESULTS_DIR),
        name="yolov8n_full_20epochs",
        exist_ok=False,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=1,
        patience=25,
        cos_lr=True,

        fraction=1.0,
        seed=SEED,
        deterministic=True,

        amp=True,
        plots=True,
        save=True,
        save_period=1,
        val=True,
        verbose=True
    )

    print("\nTraining completed!")

    run_dir = RESULTS_DIR / "yolov8n_full_20epochs"

    print("Results folder:", run_dir)
    print("Best model:", run_dir / "weights" / "best.pt")
    print("Last model:", run_dir / "weights" / "last.pt")
    print("Metrics CSV:", run_dir / "results.csv")
    print("Training plots:", run_dir / "results.png")


if __name__ == "__main__":
    main()
