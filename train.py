from ultralytics import YOLO
import os

current_path = os.getcwd()
# Cargar el modelo pre-entrenado YOLOv8m
model_path = os.path.join(os.path.dirname(__file__), 'yolov8m.pt')
model = YOLO(model_path)

# Realizar fine-tuning
results = model.train(
    data=os.path.join(current_path,'TrafficLight.yaml'),
    epochs=40,
    imgsz=640,
    batch=16,
    name='yolov8m_traffic_light',
    device=0
)

# Guardar el modelo entrenado
model.save('yolov8m_TrafficLight.pt')