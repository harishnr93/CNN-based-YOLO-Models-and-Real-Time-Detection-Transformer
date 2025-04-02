"""
Date: 01.April.2025
Author: Harish Natarajan Ravi
Email: harrish.nr@gmail.com
"""

from ultralytics import YOLO

# Load a COCO-pretrained YOLO11X model

""" 
--------------------------------------------------------------------------------------------
  Model      size        mAP-val    Speed              Speed               Params   FLOPs    
            (pixels)     50-95      CPU ONNX(ms)       T4 TensorRT10(ms)    (M)      (B)     
--------------------------------------------------------------------------------------------
YOLO11n      640         39.5       56.1  ± 0.8        1.5 ± 0.0            2.6      6.5      
--------------------------------------------------------------------------------------------
YOLO11s      640         47.0       90.0  ± 1.2        2.5 ± 0.0            9.4      21.5     
--------------------------------------------------------------------------------------------
YOLO11m      640         51.5       183.2 ± 2.0        4.7 ± 0.1            20.1     68.0    
--------------------------------------------------------------------------------------------
YOLO11l      640         53.4       238.6 ± 1.4        6.2 ± 0.1            25.3     86.9     
--------------------------------------------------------------------------------------------
YOLO11x      640         54.7       462.8 ± 6.7        11.3 ± 0.2           56.9     194.9   
--------------------------------------------------------------------------------------------
""" 

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