from PIL import Image
from ultralytics import YOLO
import cv2
import os

root_folder = os.getcwd()
model_folder = os.path.join(root_folder,"models")
model_path = os.path.join(model_folder,"best_refined.onnx")

model = YOLO(model_path)

print(model)

def yolo_find(image_path:str):
    im1 = Image.open(image_path)
    results = model.predict(source=im1, save=True) 
    boxes=[]
    confidences=[]
    for result in results:
        detected_boxes = result.boxes  # This should give you the bounding boxes
        for box in detected_boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]  # Bounding box coordinates as integers
            confidence = box.conf[0].item()  # Confidence score
            confidences.append(confidence)
            class_id = box.cls[0].item()  # Class ID

            print(f"Box: ({x1}, {y1}, {x2}, {y2}), Confidence: {confidence}, Class ID: {class_id}")

            # If you want the class name, ensure your model or dataset has a class names list
            class_name = model.names[int(class_id)]
            print(f"Detected: {class_name} with confidence {confidence}")

            boxes.append([x1, y1, x2, y2])

    return boxes,confidences

def yolo_find_Mat(image:cv2.Mat):#not working
    image=image/255
    results = model.predict(source=image, save=True) 
    boxes=[]
    confidences=[]
    for result in results:
        boxes = result.boxes  # This should give you the bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
            confidence = box.conf[0].item()  # Confidence score
            confidences.append(confidence)
            class_id = box.cls[0].item()  # Class ID

            print(f"Box: ({x1}, {y1}, {x2}, {y2}), Confidence: {confidence}, Class ID: {class_id}")

            # If you want the class name, ensure your model or dataset has a class names list
            class_name = model.names[int(class_id)]
            print(f"Detected: {class_name} with confidence {confidence}")
    boxes= [box.xyxy[0].tolist() for box in boxes]
    return boxes,confidences


if __name__ == "__main__":
    boxes,confidences = yolo_find("panneau_test2.jpg")
    img = cv2.imread("panneau_test2.jpg")
    print(boxes)
    print(confidences)
