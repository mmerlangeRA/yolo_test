from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model.info()
# Use the model
#model.train(data="coco8.yaml", epochs=3)  # train the model
model.train(data=r"C:\Users\mmerl\projects\yolo_test\datasets\tap\data.yaml", epochs=10, val=True, plots = True)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format