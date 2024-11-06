import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import torch

#add batch processing
'''
image_files = [test_image1, test_image2, test_image3]
images = [cv2.imread(img_file) for img_file in image_files]
'''
# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'  # Uncomment to force CPU usage

path_to_model = r"C:\Users\mmerl\projects\yolo_test\floutage.pt"
test_image = r'C:\Users\mmerl\projects\yolo_test\test1.jpg'

# Load the model
model = YOLO(path_to_model)
# Move the model to the desired device
model.to(device)

# Verify the device the model is on
print("Model device:", next(model.model.parameters()).device)

# Warm-up (optional but recommended)
for _ in range(5):
    _ = model.predict(test_image, device=device)

start_time = time.time()

image = cv2.imread(test_image)
end_time_load = time.time()

start_time_predict = time.time()

# Make predictions and specify the device
results = model.predict(image, device=device)#half=True

# Synchronize GPU
if device == 'cuda':
    torch.cuda.synchronize()

end_time_predict = time.time()

filePath = r'C:\Users\mmerl\projects\yolo_test\result_test.jpg'
print("Model parameters are on device:", next(model.model.parameters()).device)

for result in results:
    print("Result boxes tensor device:", result.boxes.data.device)
# Process the results and apply blurring to detected boxes
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Ensure the coordinates are within image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        # Calculate blur size and ensure it's an odd integer
        blur_size = 15  # You can adjust the blur size if needed
        if blur_size % 2 == 0:
            blur_size += 1
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue  # Skip if ROI is empty
        blurred_roi = cv2.GaussianBlur(roi, (blur_size, blur_size), 0)
        image[y1:y2, x1:x2] = blurred_roi
cv2.imwrite(filePath, image)
end_time = time.time()

print(f"Loading time: {end_time_load - start_time} seconds")
print(f"Prediction execution time: {end_time_predict - start_time_predict} seconds")
print(f"Processing execution time: {end_time - end_time_predict} seconds")
print(f"Total execution time: {end_time - start_time} seconds")
