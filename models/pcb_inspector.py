
from ultralytics import YOLO

# Load pretrained YOLO11 nano model
model = YOLO("yolo11n.pt")

# Train on PCB-MC dataset
model.train(
    data=r"D:\PROMETHEUS_DATA\PCB_MC\images\PCB-MC-A\kfold_data\fold_1\data_local.yaml",
    epochs=10,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    project="runs/pcb",
    name="prometheus_v1"
)

print("PROMETHEUS-X TRAINING COMPLETED")
