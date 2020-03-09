from detect import detector
import cv2
import time
import argparse
from sys import platform
import numpy as np

from models import *  # set ONNX_EXPORT in models.py
from yolo_tiny_utils.datasets import *
from yolo_tiny_utils.utils import *

cap = cv2.VideoCapture('2.mp4')
# Initialize

weights='best4.pt'
cfg='yolov3-tiny-1cls-se.cfg'
img_size = 416
device = torch_utils.select_device(device='0')
print(device)

# Initialize model
model = Darknet(cfg, img_size)

# Load weights
attempt_download(weights)
if weights.endswith('.pt'):  # pytorch format
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
else:  # darknet format
    load_darknet_weights(model, weights)

# Eval mode
model.to(device).eval()
while True:
    ret, frame = cap.read()
    if ret:
        print(frame.shape)
        start = time.time()
        bbox_xywh, cls_conf, cls_ids = detector(frame, device, model)
        end = time.time()
        print(bbox_xywh, cls_conf, cls_ids)
        print(end-start,1/(end-start+0.00000001))

    else:
        break

cap.release()
cv2.destroyAllWindows()





