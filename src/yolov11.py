"""
Date: 01.April.2025
Author: Harish Natarajan Ravi
Email: harrish.nr@gmail.com
"""

from ultralytics import YOLO

# Load a COCO-pretrained YOLO11X model
model_path = input("Enter the YOLO11 model (e.g., 'yolo11x.pt'): ")
model = YOLO(model_path)

# Model information
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Inference with YOLO11 model
#results = model("inference_data/BEV_data.mp4", save=True, show=True)
video_path = input("Enter the path to the video file (e.g., 'highway.mp4'): ")
results = model(video_path, save=True, show=True)