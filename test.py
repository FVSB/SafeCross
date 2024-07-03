from ultralytics import YOLO
import os

path = os.path.join(os.path.dirname(__file__), 'best.pt')
model=YOLO(path)

results = model.predict(source='dataset/images/test/',save = True, save_txt=True)
metrics = model.info(detailed=True, verbose=True)