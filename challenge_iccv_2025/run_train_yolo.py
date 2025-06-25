import os

# ✅ Desativa o WandB antes de qualquer importação do Ultralytics
os.environ["WANDB_MODE"] = "disabled"

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="/home/yasmws/Documentos/deeplearning/fisheye8k/FishEye8K/challenge_iccv_2025/data/Fisheye8k.yaml",
    epochs=100,
    batch=32,
    imgsz=640,
    device="cpu",
    task="train",
    name="yolo11n_fisheye8k32",
    save_period=10,
    save=True
)
