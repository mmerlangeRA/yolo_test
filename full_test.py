import cv2
import numpy as np
from src.yolo_detection import yolo_find
from src.opencv_detection import cv2_find
from src.matching.match_simple_pytorch import find_matching, setup_references
import os


reference_features, reference_image_paths = setup_references(r'C:\Users\mmerl\projects\yolo_test\src\matching\panneaux')



image_path = "test1.jpg"
image = cv2.imread(image_path)

boxes,confidences = yolo_find(image_path)

#boxes, confidences = cv2_find(img_path)
index = 0
for reference in boxes:
    index+=1
    x1,y1,x2,y2 = reference
    blob_image = image[y1:y2,x1:x2]

    file_name = f'blob{index}.png'
    path = os.path.join(os.getcwd(),file_name)
    cv2.imwrite(path,blob_image)
    best_match_idx = find_matching(path)
    best_match_path = reference_image_paths[best_match_idx]
    best_match_img = cv2.imread(best_match_path)
    cv2.imshow('Query Image', blob_image)
    cv2.imshow('Best Match Image', best_match_img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
