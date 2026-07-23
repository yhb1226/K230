from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.predict(source='0', show=True)   # 打开摄像头实时检测