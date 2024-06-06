import cv2
import numpy as np
from scipy.optimize import minimize

def remove_other_color(img:cv2.Mat)->cv2.Mat:
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



def preprocess_query_image(img:cv2.Mat,ref_image:cv2.Mat)->cv2.Mat:
    filtered = remove_other_color(img)
    img = cv2.bitwise_and(img, img, mask=filtered)
    return img
    gray_ref = cv2.cvtColor(ref_image,cv2.COLOR_RGB2GRAY)
    h_ref, w_ref = gray_ref.shape
    h, w = img.shape
    if h>h_ref or w>w_ref:
        max_size=max(h,w)
        ratio = h/max_size
        new_h = int(h/ratio)
        new_w = int(w/ratio)
        img = cv2.resize(img, (new_h, new_w))
    canvas = np.zeros_like(gray_ref)
    canvas[:h, :w] = img
    cv2.imshow("query",canvas)
    return canvas


def edge_detection(image:cv2.Mat):
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges

def transform_image(image, M, dsize):
    return cv2.warpPerspective(image, M, dsize, flags=cv2.INTER_LINEAR)

# Define the optimization function
def objective_function(params, query_edges_dt, ref_edges, dsize):
    M = np.array(params).reshape(3, 3)
    transformed_query_edges = transform_image(ref_edges, M, dsize)
    distance = np.sum(query_edges_dt * transformed_query_edges)
    return distance

def generate_initial_transforms(num_transforms):
    initial_transforms = []
    for _ in range(num_transforms):
        transform = np.array([
            [np.random.uniform(0.9, 1.1), np.random.uniform(-0.2, 0.2), np.random.uniform(-30, 30)],
            [np.random.uniform(-0.2, 0.2), np.random.uniform(0.9, 1.1), np.random.uniform(-30, 30)],
            [np.random.uniform(-2e-3, 2e-3), np.random.uniform(-2e-3, 2e-3), 1.0]
        ])
        initial_transforms.append(transform.flatten())
    return initial_transforms

def find_best_perspective_transform(ref_image, query_image,num_initial_transforms=100):
    ref_edges = edge_detection(ref_image)
    query_edges = edge_detection(query_image)
    
    query_edges_dt = cv2.distanceTransform(255 - query_edges, cv2.DIST_L2, 5)
    query_edges_dt = cv2.normalize(query_edges_dt, None, 0, 1.0, cv2.NORM_MINMAX)
    #cv2.imshow("Distance transform",query_edges_dt)
    dsize = (ref_image.shape[1], ref_image.shape[0])
    
    initial_transform = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    initial_params = initial_transform.flatten()

    bounds = [
        (0.9, 1.1), (-0.2, 0.2), (-30, 30),  # First row bounds
        (-0.2, 0.2), (0.9, 1.1), (-30, 30),  # Second row bounds
        (-2e-3, 2e-3), (-2e-3, 2e-3), (0.9, 1.1)  # Third row bounds
    ]
    
    best_transform = None
    best_distance = float('inf')
    
    initial_transforms = generate_initial_transforms(num_initial_transforms)
    
    for initial_params in initial_transforms:
        result = minimize(objective_function, initial_params, args=(query_edges_dt, ref_edges, dsize),
                          method='Powell', bounds=bounds)#, options={'disp': True, 'gtol': 1e-8}
        
        if result.fun < best_distance:
            best_distance = result.fun
            print(f'now best_distance is {best_distance}')
            best_transform = result.x.reshape(3, 3)
    
    return best_transform

def apply_transform_to_point(point, M):
    point_homogeneous = np.array([point[0], point[1], 1]).reshape(3, 1)
    transformed_point_homogeneous = np.dot(M, point_homogeneous)
    transformed_point = (transformed_point_homogeneous / transformed_point_homogeneous[2]).astype(int)
    return (transformed_point[0, 0], transformed_point[1, 0])

# Load images
ref_image = cv2.imread(r'C:\Users\mmerl\projects\yolo_test\src\reference.png', cv2.IMREAD_COLOR)
query_image = cv2.imread(r'C:\Users\mmerl\projects\yolo_test\src\query.png', cv2.IMREAD_COLOR)

print(ref_image.shape)
print(query_image.shape)

test_size = 129

h_ref, w_ref, _ = ref_image.shape
ref_image = cv2.resize(ref_image,(test_size,test_size))
query_image = cv2.resize(query_image,(test_size,test_size))

resized_query_img = preprocess_query_image(query_image,ref_image)

best_transform = find_best_perspective_transform(ref_image, query_image)

print("Best Perspective Transform:")
print(best_transform)

transformed_ref_image = transform_image(ref_image, best_transform, (ref_image.shape[1], ref_image.shape[0]))

# Display the images
cv2.imshow('Reference Image', ref_image)
cv2.imshow('Query Image', query_image)
#cv2.imshow('Transformed Query Image', transformed_ref_image)
alpha = 0.5  # Blending factor
print(transformed_ref_image.shape)
print(resized_query_img.shape)
blended_image = cv2.addWeighted(transformed_ref_image, alpha, resized_query_img, 1 - alpha, 0)

center_x=int(test_size/2)
center = (center_x,center_x)
print(center)
center_ref = (w_ref // 2, h_ref // 2)
top = (w_ref // 2, 0)
bottom = (w_ref // 2, h_ref -1)
print(bottom)



transformed_center = apply_transform_to_point(center_ref, best_transform)
transformed_top = apply_transform_to_point(top, best_transform)
transformed_bottom = apply_transform_to_point(bottom, best_transform)
print(transformed_bottom)
# Draw a small circle at the transformed center point on the blended image
cv2.circle(blended_image, transformed_center, 5, (0, 0, 255), 1)
cv2.circle(blended_image, transformed_top, 5, (0, 255, 0), -1)
cv2.circle(blended_image, transformed_bottom, 5, (0, 255, 0), -1)

cv2.imshow('blended_image', blended_image)
a = np.array(top)
b = np.array(bottom)
dist = np.linalg.norm(a-b)
print(f'dist is {dist}')

cv2.waitKey(0)
cv2.destroyAllWindows()


