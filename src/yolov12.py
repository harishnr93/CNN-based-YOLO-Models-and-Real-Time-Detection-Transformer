"""
Date: 02.April.2025
Author: Harish Natarajan Ravi
Email: harrish.nr@gmail.com
"""

from ultralytics import YOLO

# Load a COCO-pretrained YOLO12X model

""" 
-------------------------------------------------------------------------------------------------------
  Model      size        mAP-val     Speed               Params   FLOPs    Comparison
            (pixels)     50-95      T4 TensorRT(ms)      (M)      (B)     (mAP/Speed)
-------------------------------------------------------------------------------------------------------
YOLO12n      640         40.6         1.64               2.6      6.5      +2.1%/-9% (vs. YOLOv10n)
-------------------------------------------------------------------------------------------------------
YOLO12s      640         48.0         2.61               9.3      21.4     +0.1%/+42% (vs. RT-DETRv2)
-------------------------------------------------------------------------------------------------------
YOLO12m      640         52.5         4.86               20.2     67.5     +1.0%/-3% (vs. YOLO11m)
-------------------------------------------------------------------------------------------------------
YOLO12l      640         53.7         6.77               26.4     88.9     +0.4%/-8% (vs. YOLO11l)
-------------------------------------------------------------------------------------------------------
YOLO12x      640         55.2         11.79              59.1     199.0    +0.6%/-4% (vs. YOLO11x)
-------------------------------------------------------------------------------------------------------
""" 


model_path = input("Enter the YOLO12 model (e.g., 'yolo12x.pt'): ")
model = YOLO(model_path)

# Model information
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Inference with YOLO12 model
#results = model("inference_data/BEV_data.mp4", save=True, show=True)
video_path = input("Enter the path to the video file (e.g., 'highway.mp4'): ")
results = model(video_path, save=True, show=True)