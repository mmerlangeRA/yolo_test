# yolo_test

## Train
yolo task=detect \
mode=train \
model=yolov8s.pt \
data={dataset.location}/data.yaml \
epochs=100 \
imgsz=640

## Ressources
https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/


## Pose estimation

DenseIM ?
PoseCNN

FS6D:https://github.com/ethnhe/FS6D-PyTorch

https://github.com/liuyuan-pal/Gen6D
https://github.com/zju3dv/OnePose