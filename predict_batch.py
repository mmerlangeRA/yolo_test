import concurrent.futures
import cv2
from ultralytics import YOLO
import os
import time
import torch
from typing import List, Tuple


class ImageExtract:
    offset_x: int
    offset_y: int
    data: cv2.typing.MatLike

    def __init__(self, offset_x: int, offset_y: int, data: cv2.typing.MatLike) -> None:
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.data = data


def warm_up(model, device, img):
    # Warm-up (optional but recommended)
    for _ in range(5):
        _ = model.predict(img, device=device, verbose=False)


def pixelate_region(
    image: cv2.typing.MatLike, x1: int, y1: int, x2: int, y2: int, pixelation_level=10
) -> None:
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


def split_image_in_squares(image: cv2.typing.MatLike) -> List[ImageExtract]:
    h, w, _ = image.shape
    center = image[h // 3 : 2 * h // 3, :]
    quarter_width = w // 4
    squares = [
        ImageExtract(0, h // 3, center[:, 0:quarter_width]),
        ImageExtract(
            quarter_width, h // 3, center[:, quarter_width : quarter_width * 2]
        ),
        ImageExtract(
            quarter_width * 2, h // 3, center[:, quarter_width * 2 : quarter_width * 3]
        ),
        ImageExtract(quarter_width * 3, h // 3, center[:, quarter_width * 3 : w]),
        # ]
        # center[:, 0:quarter_width],
        # center[:, quarter_width:quarter_width*2],
        # center[:, quarter_width*2 :quarter_width*3],
        # center[:, quarter_width*3:w]
    ]
    return squares


def load_image(image_path: str) -> cv2.typing.MatLike:
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image {image_path}")
        return image_path, image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return image_path, None


def save_image(args) -> None:
    try:
        output_path, image = args
        cv2.imwrite(output_path, image)
    except Exception as e:
        print(f"Error saving image {output_path}: {e}")


def load_images(image_paths: List[str]) -> Tuple[List[cv2.typing.MatLike], List[str]]:
    images = []
    failed_images = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit image loading tasks
        future_to_image = {
            executor.submit(load_image, path): path for path in image_paths
        }

        # Process the completed futures as they finish
        for future in concurrent.futures.as_completed(future_to_image):
            image_path = future_to_image[future]
            try:
                img_path, img = future.result()
                if img is not None:
                    images.append(img)
                else:
                    failed_images.append(img_path)
            except Exception as e:
                print(f"Exception occurred while loading {image_path}: {e}")
                failed_images.append(image_path)
        return images, failed_images


def save_images(images, output_paths) -> List[str]:
    failed_saved_paths = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(save_image, (output_paths[i], images[i]))
            for i in range(len(images))
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error saving image: {e}")
                failed_saved_paths.append(output_paths[futures.index(future)])
    return failed_saved_paths


def anonymize_images_array(
    prod_images: List[cv2.typing.MatLike], pixelation_level=15, debug=False
) -> List[cv2.typing.MatLike]:
    """
    Anonymize a list of images by detecting objects and pixelating the detected regions.
    Images are modified "in place", meaning prod_images are modified and returned.

    Parameters:
        prod_images (List[cv2.typing.MatLike]): List of input images.

    Returns:
        List[cv2.typing.MatLike]: List of anonymized images.
    """
    start_time = time.time()
    all_splitted_images = []
    all_extracts: List[ImageExtract] = []
    nb_extracts_per_image = 4
    margin = 10  # let's grow the boxes a bit

    # Split each image into squares and collect them
    for idx, prod_image in enumerate(prod_images):

        # Split the image into four squares
        extracts = split_image_in_squares(prod_image)
        nb_extracts_per_image = len(extracts)
        splitted_images = [i.data for i in extracts]
        all_splitted_images.extend(splitted_images)
        all_extracts.extend(extracts)

    end_time_split = time.time()

    # Make predictions on all squares at once
    results = model.predict(
        all_splitted_images, device=device, conf=0.1, half=True, verbose=debug
    )
    all_splitted_images = None

    # Synchronize GPU
    if device == "cuda":
        torch.cuda.synchronize()

    end_time_predict = time.time()

    for image_index in range(len(prod_images)):
        prod_image = prod_images[image_index]
        h, w = prod_image.shape[:2]
        square_index = 0
        for extract_index in range(
            image_index * nb_extracts_per_image,
            (image_index + 1) * nb_extracts_per_image,
        ):
            square_index += 1
            result = results[extract_index]
            extract = all_extracts[extract_index]
            boxes = result.boxes
            if debug:
                print(
                    f"Image {image_index}, square {square_index}, nb boxes: {len(boxes)}"
                )
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Adjust coordinates to match the original image
                x1 += extract.offset_x - margin
                x2 += extract.offset_x + margin
                y1 += extract.offset_y - margin
                y2 += extract.offset_y + margin

                # Ensure the coordinates are within image boundaries
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w - 1, x2)
                y2 = min(h - 1, y2)

                # Apply pixelation to the detected region
                pixelate_region(
                    prod_image, x1, y1, x2, y2, pixelation_level=pixelation_level
                )
                if debug:
                    cv2.rectangle(prod_image, (x1, y1), (x2, y2), (0, 255, 0), 20)

    end_time = time.time()
    if debug:
        print(f"Image splitting time: {end_time_split - start_time} seconds")
        print(f"Prediction execution time: {end_time_predict - end_time_split} seconds")
        print(f"Processing time: {end_time - end_time_predict} seconds")
        print(f"Total time: {end_time - start_time} seconds")

    # Return the list of anonymized images
    return prod_images


# path model
path_to_model = r"C:\Users\mmerl\projects\yolo_test\floutage.pt"

# image infos
image_directory = r"C:\Users\mmerl\projects\yolo_test\data\logiroad"
output_dir = r"C:\Users\mmerl\projects\yolo_test\output"

# processing infos
batch_size = 16
pixelation_level = 15
failed_images_processing_paths = []
failed_images_saving_paths = []
nb_failures = 0
max_nb_failures_before_stop = 10

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device us:", device)

# Load the model
model = YOLO(path_to_model, verbose=False)
# Move the model to the desired device
model.to(device)

# Verify the device the model is on
print("Model device:", next(model.model.parameters()).device)

# warm up (optional, just to ensure speed tests are ok)
test_image_path = (
    r"C:\Users\mmerl\projects\stereo_cam\data\Photos\P5\D_P5_CAM_G_2_CUBE.png"
)
warm_up(model, device, cv2.imread(test_image_path))

# get paths of all images in folder data/logiroad
image_local_paths = os.listdir(image_directory)
image_global_paths = [
    os.path.join(image_directory, image_path) for image_path in image_local_paths
]

print(
    f"Processing all images ({len(image_global_paths)}) by batch of size: {batch_size}"
)
start_time = time.time()

# let's process image_paths by batch of batch_size images
image_paths_batches = [
    image_global_paths[i : i + batch_size]
    for i in range(0, len(image_global_paths), batch_size)
]

for batch in image_paths_batches:
    image_to_process_paths = batch
    images, failed_images_paths = load_images(image_to_process_paths)
    if len(failed_images_paths) > 0:
        nb_failures += len(failed_images_paths)
        print(
            f"Failed to load {len(failed_images_paths)} images: {failed_images_paths}"
        )
        failed_images_processing_paths.extend(failed_images_paths)
        # remove failed images from image_paths, so we don't generate wrong output paths for ex
        for failed_image_path in failed_images_paths:
            image_to_process_paths.remove(failed_image_path)
    try:
        anonymize_images_array(images, pixelation_level=pixelation_level, debug=False)
    except Exception as e:
        print(f"Error anonymizing images: {e}")
        failed_images_processing_paths.extend(image_to_process_paths)

    # Prepare output file paths
    output_file_paths = [
        os.path.join(output_dir, f"anonymized_{os.path.basename(image_path)}")
        for image_path in image_to_process_paths
    ]

    failed_images_paths = save_images(images, output_file_paths)

    if len(failed_images_paths) > 0:
        nb_failures += len(failed_images_paths)
        print(
            f"Failed to save {len(failed_images_paths)} images: {failed_images_paths}"
        )
        failed_images_saving_paths.extend(failed_images_paths)

    if nb_failures > max_nb_failures_before_stop:
        print(f"Too many failures ({nb_failures}), stopping")
        break

end_time = time.time()
if len(failed_images_processing_paths) > 0:
    print(
        f"Failed to process {len(failed_images_processing_paths)} images: {failed_images_processing_paths}"
    )
if len(failed_images_saving_paths) > 0:
    print(
        f"Failed to save {len(failed_images_saving_paths)} images: {failed_images_saving_paths}"
    )
print(f"Total execution time: {end_time - start_time} seconds")
print(
    f"Average execution time: {(end_time - start_time)/len(image_global_paths)} seconds"
)
