import cv2
import numpy as np
import os

reference_folder = 'panneaux/'
query_image_path = 'query2.jpg'

# Function to find matches and homography
def find_homography_and_matches(kp1, des1, kp2, des2):
    if des1 is None :
        print("des1 is NONE")
        return 0, None, None, []
    if des2 is None :
        print("des2 is NONE")
        return 0, None, None, []
    # Use BFMatcher to match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test as per Lowe's paper
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Minimum number of matches
    MIN_MATCH_COUNT = 10
    print(f'ref_name {len(matches)}  {len(good)}')
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        return len(good), M, matchesMask, good
    else:
        return 0, None, None, []

# Load the query image in grayscale
img_query = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
if img_query is None:
    raise ValueError("No query image")
# Initialize AKAZE detector
akaze = cv2.AKAZE_create()

# Find keypoints and descriptors for the query image
kp_query, des_query = akaze.detectAndCompute(img_query, None)
if des_query is None:
    raise ValueError("Descriptors for the query image could not be computed.")

# Initialize variables to store the best match
best_match_count = 0
best_match_image = None
best_match_homography = None
best_match_mask = None
best_match_good = None
best_reference_kp = None
best_reference_image_name = ""

# Precompute keypoints and descriptors for all reference images
reference_images = []
for filename in os.listdir(reference_folder):
    
    if filename.endswith('.jpg') or filename.endswith('.png'):
        reference_image_path = os.path.join(reference_folder, filename)
        img_ref = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
        kp_ref, des_ref = akaze.detectAndCompute(img_ref, None)
        print(f'adding {filename} with {len(kp_ref)}' )
        reference_images.append((filename, img_ref, kp_ref, des_ref))

# Iterate through all precomputed reference images
for ref_name, ref_img, kp_ref, des_ref in reference_images:
    # Find matches and homography with the current reference image
    match_count, M, matchesMask, good = find_homography_and_matches(kp_query, des_query, kp_ref, des_ref)
    
    # Update the best match if the current one has more good matches
    if match_count > best_match_count:
        best_match_count = match_count
        best_match_image = ref_img
        best_match_homography = M
        best_match_mask = matchesMask
        best_match_good = good
        best_reference_kp = kp_ref
        best_reference_image_name = ref_name

# Draw matches for the best match
if best_match_image is not None:
    h, w = img_query.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, best_match_homography)
    best_match_image = cv2.polylines(best_match_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=best_match_mask,  # draw only inliers
                       flags=2)

    img_matches = cv2.drawMatches(img_query, kp_query, best_match_image, best_reference_kp, best_match_good, None, **draw_params)

    # Show the final result
    cv2.imshow(f"Best Match with {best_reference_image_name}", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No good matches found with any reference images.")
