from multiprocessing import Pool
import concurrent.futures
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import torch
from PIL import Image
from typing import List

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'  # Uncomment to force CPU usage

def warm_up(model, device, img):
    # Warm-up (optional but recommended)
    for _ in range(5):
        _ = model.predict(img, device=device)

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
    return
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

def save_image(args):
    output_path, image = args
    cv2.imwrite(output_path, image)


path_to_model = r"C:\Users\mmerl\projects\yolo_test\floutage.pt"
image_directory = r'C:\Users\mmerl\projects\yolo_test\data\logiroad'
output_dir = r'C:\Users\mmerl\projects\yolo_test\output'
batch_size = 16

# Load the model
model = YOLO(path_to_model,verbose=False)
# Move the model to the desired device
model.to(device)
# Verify the device the model is on
print("Model device:", next(model.model.parameters()).device)

test_image_path = r"C:\Users\mmerl\projects\stereo_cam\data\Photos\P5\D_P5_CAM_G_2_CUBE.png" 
output_file_path = r'C:\Users\mmerl\projects\yolo_test\result_test_cv2.png'
#warm up
warm_up(model, device, cv2.imread(test_image_path))

def anonymize_image(prod_image:cv2.typing.MatLike, debug=False)->cv2.typing.MatLike:
    start_time = time.time()

    h, w, _ = prod_image.shape
    tier_height = h//3
    quarter_width = w // 4
    splitted_images= split_image_in_4(prod_image)
    end_time_split = time.time()

    # Make predictions and specify the device
    results = model.predict(splitted_images, device=device,conf=0.1,half=True, verbose=debug)#half=True

    # Synchronize GPU
    if device == 'cuda':
        torch.cuda.synchronize()

    end_time_predict = time.time()

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

            pixelate_region(prod_image, x1, y1, x2, y2, pixelation_level=15)
            if debug:
                print(f"nb boxes :{len(boxes)}")
                print("box",x1, y1, x2, y2)
                cv2.rectangle(prod_image, (x1, y1), (x2, y2), (0, 255, 0), 20)
        index+=1
    
    end_time = time.time()
    if debug:
        print(f"Image splitting time: {end_time_split - start_time} seconds")
        print(f"Prediction execution time: {end_time_predict - end_time_split} seconds")
        print(f"Processing and save execution time: {end_time - end_time_predict} seconds")
        print(f"Total time: {end_time - start_time} seconds")
    return 


def anonymize_images_array(prod_images: List[cv2.typing.MatLike], debug=False) -> List[cv2.typing.MatLike]:
    """
    Anonymize a list of images by detecting objects and pixelating the detected regions.

    Parameters:
        prod_images (List[cv2.typing.MatLike]): List of input images.

    Returns:
        List[cv2.typing.MatLike]: List of anonymized images.
    """
    start_time = time.time()
    all_splitted_images = []
    image_info_list = []

    # Split each image into quadrants and collect them
    for idx, prod_image in enumerate(prod_images):
        h, w, _ = prod_image.shape
        tier_height = h // 3
        quarter_width = w // 4

        # Split the image into four quadrants
        splitted_images = split_image_in_4(prod_image)
        all_splitted_images.extend(splitted_images)

        # Store information to map predictions back to the original images
        for quadrant_index in range(4):
            image_info_list.append({
                'prod_image': prod_image,
                'quarter_width': quarter_width,
                'tier_height': tier_height,
                'quadrant_index': quadrant_index,
                'image_index': idx
            })

    end_time_split = time.time()

    # Make predictions on all quadrants at once
    results = model.predict(all_splitted_images, device=device, conf=0.1, half=True,verbose=debug)

    # Synchronize GPU
    if device == 'cuda':
        torch.cuda.synchronize()

    end_time_predict = time.time()

    # Process the results and apply pixelation to detected boxes
    for result, image_info in zip(results, image_info_list):
        prod_image = image_info['prod_image']
        quarter_width = image_info['quarter_width']
        tier_height = image_info['tier_height']
        quadrant_index = image_info['quadrant_index']
        image_index = image_info['image_index']

        boxes = result.boxes
        if debug:
            print(f"Image {image_index}, Quadrant {quadrant_index}, nb boxes: {len(boxes)}")
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Adjust coordinates to match the original image
            x1 += quarter_width * quadrant_index
            x2 += quarter_width * quadrant_index
            y1 += tier_height
            y2 += tier_height

            # Ensure the coordinates are within image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(prod_image.shape[1], x2)
            y2 = min(prod_image.shape[0], y2)

            # Apply pixelation to the detected region
            pixelate_region(prod_image, x1, y1, x2, y2, pixelation_level=15)

    end_time = time.time()
    if debug:
        print(f"Image splitting time: {end_time_split - start_time} seconds")
        print(f"Prediction execution time: {end_time_predict - end_time_split} seconds")
        print(f"Processing time: {end_time - end_time_predict} seconds")
        print(f"Total time: {end_time - start_time} seconds")

    # Return the list of anonymized images
    return prod_images

# get paths of all files in folder data/logiroad
image_local_paths = os.listdir(image_directory)
image_global_paths = [os.path.join(image_directory, image_path) for image_path in image_local_paths]


# test performance on 1 image
test_image_path = image_global_paths[0]
output_file_path = os.path.join(output_dir, f"anonymized_{os.path.basename(test_image_path)}")

print(f"Test loading time with cv2 and PIL")
start1 = time.time()
test_PIL=Image.open(test_image_path)
print(test_PIL.size)
end1 = time.time()
test_cv2 =cv2.imread(test_image_path)
print(test_cv2.shape)
end2 = time.time()
print(f"Loading time PIL: {end1 - start1} seconds")
print(f"Loading time cv2: {end2 - end1} seconds")

print(f"Processing 1 single image: {test_image_path}")
start_time = time.time()
test_cv2 =cv2.imread(test_image_path)
anonymize_image(test_cv2)
cv2.imwrite(output_file_path, test_cv2)
end_time = time.time()
print(f"Total execution time: {end_time - start_time} seconds")

print(f"Processing all images ({len(image_global_paths)}) by batch of size: {batch_size}")
start_time = time.time()
#let's process image_paths by batch of 4 images
image_paths_batches = [image_global_paths[i:i + batch_size] for i in range(0, len(image_global_paths), batch_size)]

for batch in image_paths_batches:
    image_paths = batch
    with concurrent.futures.ThreadPoolExecutor() as executor:
        images = list(executor.map(cv2.imread, image_paths))

    anonymize_images_array(images)

    # Prepare output file paths
    output_file_paths = [
        os.path.join(output_dir, f"anonymized_{os.path.basename(image_path)}")
        for image_path in image_paths
    ]

    # Prepare the arguments for saving images
    save_args = zip(output_file_paths, images)

    # Save images in parallel using futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(save_image, save_args)


end_time = time.time()
print(f"Total execution time: {end_time - start_time} seconds")
print(f"Average execution time: {(end_time - start_time)/len(image_global_paths)} seconds")
