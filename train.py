from multiprocessing import freeze_support
from ultralytics import YOLO


def train_model():
    # Load a model
    #model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model.info()
    # Use the model
    #model.train(data="coco8.yaml", epochs=3)  # train the model
    model.train(data=r"C:\Users\mmerl\projects\yolo_test\datasets\panneaux\data.yaml", epochs=150, val=True, plots = True, batch=32)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    path = model.export(format="onnx")  # export the model to ONNX format

if __name__ == '__main__':
    freeze_support()
    train_model()