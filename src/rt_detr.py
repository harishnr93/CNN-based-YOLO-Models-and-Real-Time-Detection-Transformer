"""
Date: 26.March.2025
Author: Harish Natarajan Ravi
Email: harrish.nr@gmail.com
"""

from ultralytics import RTDETR

# Load pre-trained model

""" 
-------------------------------------------------------------------------------------
  Model Type            Pre-trained Weights   Total Layers    Params(M)     GFLOPS           
-------------------------------------------------------------------------------------
 RT-DETR Large                rtdetr-l.pt         449           32.97       108.3
-------------------------------------------------------------------------------------
 RT-DETR Extra-Large          rtdetr-x.pt         567           67.47       232.7
-------------------------------------------------------------------------------------
"""
model_path = input("Enter the RT-DETR model (e.g., 'rtdetr-x.pt'): ")
model = RTDETR(model_path)

# Model information
model.info()


# Inference with RT-DETR model
# results = model("inference_data\BEV_data.mp4", show=True, save=True)
video_path = input("Enter the path to the video file (e.g., 'highway.mp4'): ")
results = model(video_path, show=True, save=True)