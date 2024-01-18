from ultralytics import YOLO

model = YOLO('yolov8s.pt')

model.train(
    data = '/home/huemorgen/DefectDetection/datasets/NEU_DET/yolo_format/data.yaml',
    epochs = 1,
    imgsz = 640,
    optimizer = 'Adam',
    seed = 0,
    lr0 = 0.001,
    batch=4,
)

metrics = model.val()