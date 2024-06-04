import cv2
import numpy as np
from scipy.optimize import differential_evolution

# Load images
img_ref = cv2.imread(r'C:\Users\mmerl\projects\yolo_test\src\matching\panneaux\France_road_sign_B21-2.svg.png', cv2.IMREAD_GRAYSCALE)
img_target = cv2.imread(r'C:\Users\mmerl\projects\yolo_test\blob1.png', cv2.IMREAD_GRAYSCALE)

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

# Convert edges to float32
edges_reference = edges_reference.astype(np.float32) / 255.0
dist_transform_target = dist_transform_target.astype(np.float32)

# Define the optimization function
def edge_distance(params):
    a, b, tx, c, d, ty = params
    # Construct the affine transformation matrix
    affine_matrix = np.array([[a, b, tx], [c, d, ty]], dtype=np.float32)
    h, w = resized_img_ref.shape[:2]
    # Warp edges_reference to the coordinate space of edges_target
    warped_edges_reference = cv2.warpAffine(edges_reference, affine_matrix, (w, h))
    # Compute distance using the distance transform
    distance = np.sum(dist_transform_target * warped_edges_reference)
    return distance

# Define bounds for the differential evolution optimizer
bounds = [
    (0.7, 1.3),  # a
    (-0.3, 0.3),  # b
    (-20, 20),   # tx
    (-0.3, 0.3),  # c
    (0.7, 1.3),  # d
    (-20, 20)    # ty
]

# Use differential evolution to find the optimal parameters
result = differential_evolution(edge_distance, bounds, strategy='best1bin', maxiter=1000, popsize=15, tol=1e-6, mutation=(0.5, 1), recombination=0.7, disp=True)

# Extract optimized parameters
a_opt, b_opt, tx_opt, c_opt, d_opt, ty_opt = result.x

# Construct the optimized affine transformation matrix
affine_matrix_opt = np.array([[a_opt, b_opt, tx_opt], [c_opt, d_opt, ty_opt]], dtype=np.float32)

# Apply the optimized affine transformation to img_ref
img_ref_warped = cv2.warpAffine(resized_img_ref, affine_matrix_opt, (w1_r, h1_r))

# Display the results
cv2.imshow('Target Image', canvas)
cv2.imshow('Warped Reference Image', img_ref_warped)
print(affine_matrix_opt)
cv2.waitKey(0)
cv2.destroyAllWindows()
