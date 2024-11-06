from multiprocessing import freeze_support
from ultralytics import YOLO
import torch
import os
print("cuda available",torch.cuda.is_available())

def train_model():
    # Load a model
    #model = YOLO("yolov8n.yaml")  # build a new model from scratch
    #model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11n.pt")
    model.info()
    # Use the model
    #model.train(data="coco8.yaml", epochs=3)  # train the model
    data_yaml_path = r"C:\Users\mmerl\projects\yolo_test\datasets\faces_plates\data.yaml"
    if not os.path.exists(data_yaml_path):
        raise ValueError(f"Data YAML file not found at {data_yaml_path}")
    model.train(data=data_yaml_path, epochs=300, val=True, plots = True, batch=32)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    
    path = model.export(format="onnx")  # export the model to ONNX format
    print(f"saving model at {path}")
    results = model(r'C:\Users\mmerl\projects\stereo_cam\data\Photos\P5\D_P5_CAM_D_2_EAC.png')  # predict on an image
    
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        filePath = r'C:\Users\mmerl\projects\yolo_test\D_P5_CAM_D_2_EAC.jpg'
        result.save(filename=filePath)  # save to disk

if __name__ == '__main__':
    freeze_support()
    train_model()