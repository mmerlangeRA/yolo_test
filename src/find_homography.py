import cv2
import numpy as np
from scipy.optimize import minimize

# Load images
img1 = cv2.imread(r'C:\Users\mmerl\projects\yolo_test\src\matching\panneaux\France_road_sign_B21-2.svg.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r'C:\Users\mmerl\projects\yolo_test\blob1.png', cv2.IMREAD_GRAYSCALE)

# Get sizes of the images
h1, w1 = img1.shape
h2, w2 = img2.shape

# Resize img1 to match the larger dimension of img2
max_size_img2 = max(h2, w2)
resized_img1 = cv2.resize(img1, (max_size_img2, max_size_img2))
h1_r, w1_r = resized_img1.shape

# Create a canvas for img2 with the size of resized_img1
canvas = np.zeros_like(resized_img1)
canvas[:h2, :w2] = img2

# Preprocess images to reduce noise
img1_blur = cv2.GaussianBlur(resized_img1, (5, 5), 0)
img2_blur = cv2.GaussianBlur(canvas, (5, 5), 0)

# Detect edges with adjusted thresholds
edges1 = cv2.Canny(img1_blur, 50, 150)
edges2 = cv2.Canny(img2_blur, 50, 150)

# Apply morphological operations to clean up the edges
kernel = np.ones((3, 3), np.uint8)
edges1 = cv2.morphologyEx(edges1, cv2.MORPH_CLOSE, kernel)
edges2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernel)
cv2.imshow('edges2', edges2)

#inverted_edges = cv2.invert(edges2)

cv2.imshow('inverted_edges', edges2)
# Compute the distance transform of the edges of img2
dist_transform = cv2.distanceTransform(255 - edges2, cv2.DIST_L2, 5)
dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
cv2.imshow('dist_transform', dist_transform)
# Convert edges to float32
edges1 = edges1.astype(np.float32)
dist_transform = dist_transform.astype(np.float32)


#cv2.imshow('edge1', edges1)


# Define the optimization function
def edge_distance(params):
    a, b, tx, c, d, ty = params
    # Construct the affine transformation matrix
    affine_matrix = np.array([[a, b, tx], [c, d, ty]], dtype=np.float32)
    
    h, w = resized_img1.shape[:2]  # Use img1's dimensions
    # Warp edges1 to the coordinate space of edges2
    warped_edges1 = cv2.warpAffine(edges1, affine_matrix, (w, h))
    # Compute distance using the distance transform
    distance = np.sum(dist_transform * (warped_edges1 / 255.0))
    print(affine_matrix)
    print(distance)
    return distance

# Initialize parameters with reasonable values for an affine transformation
initial_params = np.array([1.1, 0, 0, 0, 1, 0]) 

# Define bounds for small transformations
skew= 0.2
t = 15
bounds = [
    (1-skew, 1+skew),  # a
    (-skew, skew),  # b
    (-t, t),  # tx
    (-skew, skew),  # c
    (1-skew, 1+skew),  # d
    (-t, t)   # ty
]

# Optimize the parameters using L-BFGS-B method with bounds
result = minimize(edge_distance, initial_params, method='L-BFGS-B', bounds=bounds, options={'disp': True, 'tol': 1e-6})

# Extract optimized parameters
a_opt, b_opt, tx_opt, c_opt, d_opt, ty_opt = result.x

# Construct the optimized affine transformation matrix
affine_matrix_opt = np.array([[a_opt, b_opt, tx_opt], [c_opt, d_opt, ty_opt]], dtype=np.float32)

# Apply the optimized affine transformation to img1
img1_warped = cv2.warpAffine(resized_img1, affine_matrix_opt, (w1_r, h1_r))

# Display the results
cv2.imshow('Target Image', canvas)
cv2.imshow('Warped Image', img1_warped)
print(affine_matrix_opt)
cv2.waitKey(0)
cv2.destroyAllWindows()
