"""
Date: 26.March.2025
Author: Harish Natarajan Ravi
Email: harrish.nr@gmail.com
"""

from ultralytics import RTDETR

# Load pre-trained model
model_path = input("Enter the RT-DETR model (e.g., 'rtdetr-l.pt'): ")
model = RTDETR('rtdetr-l.pt')

# Model information
model.info()


# Inference with RT-DETR model
# results = model("inference_data\BEV_data.mp4", show=True, save=True)
video_path = input("Enter the path to the video file (e.g., 'highway.mp4'): ")
results = model(video_path, show=True, save=True)