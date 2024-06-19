import numpy as np

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections

MIN_CONFIDENCE = 0.25
# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
# load model
model = YOLO(model_path, task="detect")

# inference
def output_bb(img, confidence = MIN_CONFIDENCE): 
    output = model(img, verbose=False)[0]

    cleaned_boxes = []
    for box in output.boxes.data:
        if box[4] > confidence:
            cleaned_boxes.append([
                int(box[0]), # x1
                int(box[1]), # y1
                int(box[2]), # x2
                int(box[3]), # y2
            ])


    return np.array(cleaned_boxes)