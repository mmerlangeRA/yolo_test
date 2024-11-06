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

def warm_up(model, device, img):
    # Warm-up (optional but recommended)
    for _ in range(5):
        _ = model.predict(img, device=device)

#test_image = r'C:\Users\mmerl\projects\yolo_test\test1.jpg'
import cv2
import numpy as np

def pixelate_region(image, x1, y1, x2, y2, pixelation_level=10):
    """
    Apply pixelation to a rectangular region in an image.

    Parameters:
        image (numpy.ndarray): The input image.
        x1, y1 (int): Top-left coordinates of the rectangle.
        x2, y2 (int): Bottom-right coordinates of the rectangle.
        pixelation_level (int): Level of pixelation (higher value means more pixelated).

    Returns:
        numpy.ndarray: The image with the pixelated region.
    """
    # Ensure coordinates are within the image boundaries
    x1 = max(0, min(x1, image.shape[1]))
    x2 = max(0, min(x2, image.shape[1]))
    y1 = max(0, min(y1, image.shape[0]))
    y2 = max(0, min(y2, image.shape[0]))
    
    # Extract the region of interest (ROI)
    roi = image[y1:y2, x1:x2]

    # Check if ROI is valid
    if roi.size == 0:
        return image  # Return original image if ROI is empty or invalid

    # Get the size of the ROI
    height, width = roi.shape[:2]

    # Determine the size to resize to
    temp_height = max(1, height // pixelation_level)
    temp_width = max(1, width // pixelation_level)

    # Resize the ROI to the smaller size
    temp = cv2.resize(roi, (temp_width, temp_height), interpolation=cv2.INTER_LINEAR)

    # Resize back to the original size using nearest neighbor interpolation
    pixelated_roi = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

    # Replace the ROI in the original image
    image[y1:y2, x1:x2] = pixelated_roi

    return image


def split_image_in_4(image):
    h, w, _ = image.shape
    center = image[h//3:2* h//3, :]
    quarter_width = w // 4
    quadrants = [
        center[:, 0:quarter_width],  
        center[:, quarter_width:quarter_width*2],  
        center[:, quarter_width*2 :quarter_width*3],  
        center[:, quarter_width*3:w]
    ]
    return quadrants

# Load the model
path_to_model = r"C:\Users\mmerl\projects\yolo_test\floutage.pt"
model = YOLO(path_to_model)
# Move the model to the desired device
model.to(device)

# Verify the device the model is on
print("Model device:", next(model.model.parameters()).device)

test_image = r"C:\Users\mmerl\projects\stereo_cam\data\Photos\P5\D_P5_CAM_G_2_CUBE.png" 

#warm up
warm_up(model, device, cv2.imread(test_image))

start_time = time.time()
prod_image=cv2.imread(test_image)
h, w, _ = prod_image.shape
tier_height = h//3
quarter_width = w // 4
splitted_images= split_image_in_4(prod_image)
end_time_load = time.time()

start_time_predict = time.time()

# Make predictions and specify the device
results = model.predict(splitted_images, device=device,conf=0.1)#half=True

# Synchronize GPU
if device == 'cuda':
    torch.cuda.synchronize()

end_time_predict = time.time()

output_file_path = r'C:\Users\mmerl\projects\yolo_test\result_test3.jpg'
#print("Model parameters are on device:", next(model.model.parameters()).device)

# Process the results and apply blurring to detected boxes
index=0
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # Ensure the coordinates are within image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(prod_image.shape[1], x2)
        y2 = min(prod_image.shape[0], y2)
        x1+=quarter_width*index
        x2+=quarter_width*index
        y1+=tier_height
        y2+=tier_height

        prod_image = pixelate_region(prod_image, x1, y1, x2, y2, pixelation_level=15)
        #print(x1, y1, x2, y2)
        #cv2.rectangle(prod_image, (x1, y1), (x2, y2), (0, 255, 0), 20)
    index+=1

cv2.imwrite(output_file_path, prod_image)
end_time = time.time()

print(f"Loading time: {end_time_load - start_time} seconds")
print(f"Prediction execution time: {end_time_predict - start_time_predict} seconds")
print(f"Processing and save execution time: {end_time - end_time_predict} seconds")
print(f"Total time: {end_time - start_time} seconds")
