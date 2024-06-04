import cv2
import numpy as np
from scipy.optimize import minimize

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

# Load images
img_ref = cv2.imread(r'C:\Users\mmerl\projects\yolo_test\src\matching\panneaux\France_road_sign_AB3.svg.png', cv2.IMREAD_GRAYSCALE)
img_target = cv2.imread(r'C:\Users\mmerl\projects\yolo_test\blob5.png', cv2.IMREAD_COLOR)

""" binary_image = preprocess_image(img_target)
binary_image = removeSmallComponents(binary_image, 200)
binary_image = cv2.bitwise_and(binary_image, binary_image, mask=remove_other_color(img_target)) """
filtered = remove_other_color(img_target)

img_target = cv2.bitwise_and(img_target, img_target, mask=filtered)

img_target = cv2.cvtColor(img_target,cv2.COLOR_RGB2GRAY)

# Get sizes of the images
h_ref, w_ref = img_ref.shape
h_target, w_target = img_target.shape

# Resize img_ref to match the larger dimension of img_target
max_size_img_target = max(h_target, w_target)
resized_img_ref = cv2.resize(img_ref, (max_size_img_target, max_size_img_target))
h1_r, w1_r = resized_img_ref.shape

# Create a canvas for img_target with the size of resized_img_ref
canvas = np.zeros_like(resized_img_ref)
canvas[:h_target, :w_target] = img_target

# Preprocess images to reduce noise
img_ref_blur = cv2.GaussianBlur(resized_img_ref, (5, 5), 0)
img_target_blur = cv2.GaussianBlur(canvas, (5, 5), 0)

# Detect edges with adjusted thresholds
edges_reference = cv2.Canny(img_ref_blur, 50, 150)
edges_target = cv2.Canny(img_target_blur, 50, 150)

# Apply morphological operations to clean up the edges
kernel = np.ones((3, 3), np.uint8)
edges_reference = cv2.morphologyEx(edges_reference, cv2.MORPH_CLOSE, kernel)
edges_target = cv2.morphologyEx(edges_target, cv2.MORPH_CLOSE, kernel)

# Compute the distance transform of the edges of img_target
dist_transform_target = cv2.distanceTransform(255 - edges_target, cv2.DIST_L2, 5)
dist_transform_target = cv2.normalize(dist_transform_target, None, 0, 1.0, cv2.NORM_MINMAX)

cv2.imshow("Distance transform",dist_transform_target)
# Convert edges to float32
edges_reference = edges_reference.astype(np.float32) / 255.0
dist_transform_target = dist_transform_target.astype(np.float32)

# Define the optimization function
def edge_distance(params):
    a, b, tx, c, d, ty = params
    a *= factorR
    b *= factorR
    c *= factorR
    d *= factorR
    tx *= factorT
    ty *= factorT
   # print(params)
    # Construct the affine transformation matrix
    affine_matrix = np.array([[a, b, tx], [c, d, ty]], dtype=np.float32)
    h, w = resized_img_ref.shape[:2]
    # Warp edges_reference to the coordinate space of edges_target
    warped_edges_reference = cv2.warpAffine(edges_reference, affine_matrix, (w, h))
    # Compute distance using the distance transform
    distance = np.sum(dist_transform_target * warped_edges_reference)
    #print(distance)
    return distance

# Define a callback function to print progress during optimization
def callback(params):
    print(f"Current params: {params}")


factorR = 100000000
factorT = 10000000
# Initialize parameters with reasonable values for an affine transformation
initial_params_array = [
    np.array([1./factorR,0.0, 0., 0.0, 1./factorR, 0.]),
    np.array([0.95/factorR,0.0, 0., 1./factorT, 0.95/factorR, 1./factorT]),
    np.array([0.9/factorR,0.0, 0., 0.0, 0.9/factorR, 0.]),
]

# Define bounds for small transformations
skew = 0.2/factorR
t = 10/factorT
bounds = [
    (1./factorR - skew, 1./factorR + skew),  # a
    (-skew, skew),         # b
    (-t, t),               # tx
    (-skew, skew),         # c
    (1./factorR - skew, 1./factorR + skew),  # d
    (-t, t)                # ty
]

print("bounds")
print(bounds)

print(edge_distance([1./factorR,0.00, 0.00, 0.0, 1./factorR, 0.00]))
print(edge_distance([1./factorR,0.001, 0.001, 0.0, 1./factorR, 0.001]))
print([1./factorR,0.0, 0, 0.0, 1./factorR, 0.])
#print(edge_distance([1.001, 0.0, 0, 0.0, 1.0, 0]))

# Optimize the parameters using L-BFGS-B method with bounds
min_dist = float("inf")
best_result = None
for initial_params in initial_params_array:
    result_tmp = minimize(edge_distance, initial_params, method='L-BFGS-B', bounds=bounds, options={'disp': True,'gtol': 1e-8}, callback=callback)
    if result_tmp.fun < min_dist:
        min_dist = result_tmp.fun
        print(f'min_dist is now {min_dist}')
        best_result = result_tmp

# Extract optimized parameters
if best_result:
    a_opt, b_opt, tx_opt, c_opt, d_opt, ty_opt = best_result.x
    print(f"edge {edge_distance([a_opt, b_opt, tx_opt,c_opt, d_opt, ty_opt])}")
    a_opt = a_opt*factorR
    b_opt = b_opt*factorR
    c_opt = c_opt*factorR
    d_opt = d_opt*factorR
    tx_opt= tx_opt*factorT
    ty_opt = ty_opt * factorT
    
    # Construct the optimized affine transformation matrix
    affine_matrix_opt = np.array([[a_opt, b_opt, tx_opt], [c_opt, d_opt, ty_opt]], dtype=np.float32)
    print(affine_matrix_opt)

    # Apply the optimized affine transformation to img_ref
    img_ref_warped = cv2.warpAffine(resized_img_ref, affine_matrix_opt, (w1_r, h1_r))

    # Display the results
    #cv2.imshow('Target Image', canvas)
    cv2.imshow('Warped Reference Image', img_ref_warped)
    print(affine_matrix_opt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Optimization failed to converge.")

