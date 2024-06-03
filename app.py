import cv2
import numpy as np
from src.yolo_detection import yolo_find
from src.opencv_detection import cv2_find
from src.matching.match_simple_pytorch import find_matching, setup_references
import os
import streamlit as st


@st.cache_data
def set_references():
    print("set_references")
    global reference_features, reference_image_paths
    reference_features, reference_image_paths = setup_references(r'C:\Users\mmerl\projects\yolo_test\src\matching\panneaux')
    return reference_features, reference_image_paths

reference_features, reference_image_paths = set_references()


st.title("Image Processing App")


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

root_folder = os.getcwd()
tmp_folder = os.path.join(root_folder,"tmp")
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_to_show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_path =  os.path.join(tmp_folder,"test.jpg")
    cv2.imwrite(image_path,image)
    st.image(image_to_show, caption='Uploaded Image', use_column_width=True)

    st.write("")
    st.write("Choose a processing algorithm")

    option = st.selectbox(
        'Processing Algorithm',
        ('OpenCV', 'YOLO')
    )

    if st.button('Process'):
        if option == 'OpenCV':
            boxes,confidences = cv2_find(image_path)
        elif option == 'YOLO':
            boxes,confidences = yolo_find(image_path)
        index = 0
        print(boxes)
        for reference in boxes:
            index+=1
            x1,y1,x2,y2 = reference
            blob_image = image[y1:y2,x1:x2]

            file_name = f'blob{index}.png'
            blob_image_path = os.path.join(tmp_folder,file_name)
            print(f"found blob, saving at {blob_image_path}")
            cv2.imwrite(blob_image_path,blob_image)
            best_match_idx = find_matching(blob_image_path)
            print(f'best_match_idx is {best_match_idx} wheres {len(reference_image_paths)}')
            best_match_path = reference_image_paths[best_match_idx]
            best_match_img = cv2.imread(best_match_path)
            st.image(best_match_img, caption='Found Image', use_column_width=True, channels="BGR")

