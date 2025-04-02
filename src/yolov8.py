"""
Date: 30.March.2025
Author: Harish Natarajan Ravi
Email: harrish.nr@gmail.com
"""

from ultralytics import YOLO

# Load a COCO-pretrained YOLOV8X model

""" 
----------------------------------------------------------------------------------------------
  Model      size        mAP-val    Speed              Speed                 Params   FLOPs    
            (pixels)     50-95      CPU ONNX(ms)       A100 TensorRT(ms)      (M)      (B)     
----------------------------------------------------------------------------------------------
YOLOv8n      640         37.3         80.4               0.99                 3.2      8.7      
----------------------------------------------------------------------------------------------
YOLOv81s      640        44.9         128.4              1.20                 11.2     28.6       
----------------------------------------------------------------------------------------------
YOLOv8m      640         50.2         234.7              1.83                 25.9     78.9     
----------------------------------------------------------------------------------------------
YOLOv8l      640         52.9         375.2              2.39                 43.7     165.2       
----------------------------------------------------------------------------------------------
YOLOv8x      640         53.9         479.1              3.53                 68.2     257.8    
----------------------------------------------------------------------------------------------
""" 

model_path = input("Enter the YOLO8 model (e.g., 'yolov8x.pt'): ")
model = YOLO(model_path)

# Model information
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Inference with YOLO8 model
#results = model("inference_data/BEV_data.mp4", save=True, show=True)
video_path = input("Enter the path to the video file (e.g., 'highway.mp4'): ")
results = model(video_path, save=True, show=True)