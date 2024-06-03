import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import os

# Function to detect blobs in the image
def detect_blobs(image):
    # Convert the image to grayscale
    gray_image = image#cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set up the SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    print(params)

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1000
    params.maxArea = 100000


    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(gray_image)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return im_with_keypoints, keypoints



# Clean all previous file
def clean_images():
    file_list = os.listdir('./')
    for file_name in file_list:
        if '.png' in file_name:
            os.remove(file_name)

### Preprocess image
def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(img_hist_equalized))
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

def LaplacianOfGaussian(image):
    LoG_image = cv2.GaussianBlur(image, (3,3), 0)           # parameter
    gray = cv2.cvtColor(LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian(gray, cv2.CV_8U, 3, 3, 2)       # parameter
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image

def binarization(image):
    thresh = cv2.threshold(image, 32, 255, cv2.THRESH_BINARY)[1]
    return thresh

def preprocess_image(image):
    image = constrastLimit(image)
    image = LaplacianOfGaussian(image)
    image = binarization(image)
    return image

# Find Signs
def removeSmallComponents(image, threshold):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    img2 = np.zeros((output.shape), dtype=np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

def findContour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def contourIsSign(perimeter, centroid, threshold):
    result = []
    for p in perimeter:
        p = p[0]
        distance = sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result]
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold:
        return True, max_value + 2
    else:
        return False, max_value + 2

def cropContour(image, center, max_distance):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(center[0] - max_distance), 0])
    bottom = min([int(center[0] + max_distance + 1), height - 1])
    left = max([int(center[1] - max_distance), 0])
    right = min([int(center[1] + max_distance + 1), width - 1])
    print(left, right, top, bottom)
    return image[left:right, top:bottom]

def cropSign(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height - 1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width - 1])
    return image[top:bottom, left:right]

def findLargestSign(image, contours, threshold, distance_theshold):
    max_distance = 0
    coordinate = None
    sign = None
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = contourIsSign(c, [cX, cY], 1 - threshold)
        if is_sign and distance > max_distance and distance > distance_theshold:
            max_distance = distance
            coordinate = np.reshape(c, [-1, 2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis=0)
            coordinate = [(left - 2, top - 2), (right + 3, bottom + 1)]
            sign = cropSign(image, coordinate)
    return sign, coordinate

def findSigns(image, contours, threshold, distance_theshold):
    signs = []
    coordinates = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, max_distance = contourIsSign(c, [cX, cY], 1 - threshold)
        if is_sign and max_distance > distance_theshold:
            sign = cropContour(image, [cX, cY], max_distance)
            signs.append(sign)
            coordinate = np.reshape(c, [-1, 2])
            top, left = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis=0)
            coordinates.append([(top - 2, left - 2), (right + 1, bottom + 1)])
    return signs, coordinates

def localization_simple(image, min_size_components, similitary_contour_with_circle):
    original_image = image.copy()
    binary_image = preprocess_image(image)
    binary_image = removeSmallComponents(binary_image, min_size_components)
    binary_image = cv2.bitwise_and(binary_image, binary_image, mask=remove_other_color(image))

    contours = findContour(binary_image)
    sign, coordinate = findLargestSign(original_image, contours, similitary_contour_with_circle, 15)
    return sign, coordinate

def remove_line(img):
    gray = img.copy()
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    minLineLength = 5
    maxLineGap = 3
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, minLineLength, maxLineGap)
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(mask, (x1, y1), (x2, y2), (0, 0, 0), 2)
    return cv2.bitwise_and(img, img, mask=mask)

def remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([255, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_white = np.array([100, 0, 0], dtype=np.uint8)
    upper_white = np.array([250, 90, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([170, 150, 50], dtype=np.uint8)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    mask_1 = cv2.bitwise_or(mask_blue, mask_white)
    mask = cv2.bitwise_or(mask_1, mask_black)
    return mask_blue


def cv2_find(image_path:str):
    image = cv2.imread(image_path)
    #processed_img = remove_other_color(im2)

    binary_image = preprocess_image(image)
    binary_image = removeSmallComponents(binary_image, 200)
    binary_image = cv2.bitwise_and(binary_image, binary_image, mask=remove_other_color(image))
    filtered = remove_other_color(image)
    #cv2.imshow("color filter", filtered)

    kernel = np.ones((4,4),np.uint8)
    erosion = cv2.erode(filtered,kernel,iterations = 1)
    #cv2.imshow("erosion", erosion)
    dilate = cv2.dilate(erosion,kernel,iterations = 1)
    #cv2.imshow("dilated", dilate)

    kernel = np.ones((40,40),np.uint8)
    dilate = cv2.dilate(dilate,kernel,iterations = 1)
    #cv2.imshow("dilated_2", dilate)

    erosion = cv2.erode(dilate,kernel,iterations = 1)
    #cv2.imshow("erosion-2", erosion)
    # Detect blobs
    blob_image, keypoints = detect_blobs(erosion)
    # Display the keypoints on the image
    #cv2.imshow("Blobs", blob_image)

    blob_image, keypoints = detect_blobs(cv2.bitwise_not(erosion))
    cv2.imshow("inverted Blobs", blob_image)

    height, width = image.shape[:2]
    print("Image Dimensions: ", width, "x", height)

    # Print keypoint details
    index=0
    references =[]
    for keypoint in keypoints:
        index+=1
        if index>0:
            x = keypoint.pt[0]
            y = keypoint.pt[1]
            size = keypoint.size
            print(f"Blob detected at ({x}, {y}) with size {size}")
            size = size*1.2
            x1=max(0,int(x-size/2.))
            x2=min(width-1,int(x+size/2.))
            y1=max(int(y-size/2.),0)
            y2=min(height-1,int(y+size/2.))
            references.append([x1,y1,x2,y2])
    return references,[]

'''
sign, coordinates = localization_simple(im2, 200, 0.5)
if sign is not None:
    cv2.imshow("Detected Sign", sign)
    cv2.waitKey(0)
else:
    print("No sign detected")
'''
