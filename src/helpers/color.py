import cv2
import numpy as np
from matplotlib import colors

def convert_image_to_hsv(image:cv2.Mat)->cv2.Mat:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

def load_and_convert_image(image_path):
    image = cv2.imread(image_path)
    return convert_image_to_hsv(image)

def reshape_image(image):
    pixels = image.reshape((-1, 3))
    return pixels

def apply_kmeans(pixels, k=2):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return labels, centers

def extract_main_colors(centers):
    centers = np.uint8(centers)
    main_colors = [tuple(color) for color in centers]
    return main_colors

def hsv_to_rgb(hsv_colors):
    rgb_colors = [colors.hsv_to_rgb([h/179, s/255, v/255]) for h, s, v in hsv_colors]
    rgb_colors = [tuple((np.array(color) * 255).astype(int)) for color in rgb_colors]
    return rgb_colors

def create_mask(image, color, lower_bound=10, upper_bound=10):
    lower = np.array([max(color[0] - lower_bound, 0), max(color[1] - lower_bound, 0), max(color[2] - lower_bound, 0)])
    upper = np.array([min(color[0] + upper_bound, 179), min(color[1] + upper_bound, 255), min(color[2] + upper_bound, 255)])
    mask = cv2.inRange(image, lower, upper)
    return mask

def combine_masks(masks):
    combined_mask = np.zeros_like(masks[0])
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    return combined_mask

def erode_mask(mask, kernel_size=5, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
    return eroded_mask

def morphological_skeleton(binary_image):
    # Ensure the input is a binary image
    binary_image = binary_image.astype(np.uint8)
    skeleton = np.zeros(binary_image.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        # Erode the image
        eroded = cv2.erode(binary_image, element)
        # Dilate the eroded image
        temp = cv2.dilate(eroded, element)
        # Subtract the dilated image from the original image
        temp = cv2.subtract(binary_image, temp)
        # Bitwise OR to add the temporary image to the skeleton
        skeleton = cv2.bitwise_or(skeleton, temp)
        # Update the image for the next iteration
        binary_image = eroded.copy()

        # If the image is completely eroded, stop the loop
        if cv2.countNonZero(binary_image) == 0:
            break

    return skeleton

def process_image(image_path, main_colors, kernel_size=5, iterations=1):
    image = cv2.imread(image_path)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    hsv_image = convert_image_to_hsv(image)
    masks = [create_mask(hsv_image, color,10,10) for color in main_colors]
    combined_mask = combine_masks(masks)
    skeleton = morphological_skeleton(combined_mask)
    eroded_mask = erode_mask(combined_mask, kernel_size=kernel_size, iterations=iterations)
    return combined_mask, eroded_mask,skeleton

if __name__ == "__main__":
    # Load images
    image_path = r'C:\Users\mmerl\projects\yolo_test\src\reference.png'
    hsv_image = load_and_convert_image(image_path)
    pixels = reshape_image(hsv_image)
    labels, centers = apply_kmeans(pixels, k=2)
    main_colors = extract_main_colors(centers)
    main_colors_rgb = hsv_to_rgb(main_colors)
    print("Main colors in HSV:", main_colors)
    print("Main colors in RGB:", main_colors_rgb)
    combined_mask, eroded_mask,skeleton = process_image(image_path, main_colors, kernel_size=3, iterations=1)
    cv2.imshow('Combined Mask', combined_mask)
    cv2.imshow('Eroded Mask', eroded_mask)
    cv2.imshow('skeleton', skeleton)
    cv2.waitKey(0)