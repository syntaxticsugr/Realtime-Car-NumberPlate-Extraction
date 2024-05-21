from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

model.train(data="src/yolo_config.yaml", epochs=10)

metrics = model.val()

path = model.export(format="onnx")

print(path)
