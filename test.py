""" import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX("yolov8m-face.onnx") """

import cv2
from PIL import Image
from ultralytics import YOLO

model = YOLO("best10.onnx")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
print(model)
# from PIL
im1 = Image.open("tap2.jpg")
results = model.predict(source=im1, save=True)  # save plotted images

# from ndarray
im2 = cv2.imread("tap3.jpg")
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

im3 = Image.open("tap4.jpg")
results = model.predict(source=im3, save=True)  # save plotted images

im4 = Image.open("tap5.jpg")
results = model.predict(source=im4, save=True)  # save plotted images