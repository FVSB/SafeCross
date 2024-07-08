import cv2
from PIL import Image
from ultralytics import YOLO, solutions
import IPython.display as dp

def decide(img_path:str,conf:float=0.75)->bool:
    
    # Los pesos
    weigths_path='/home/SafeCross/weigths'
    crosswalk_weight=f'{weigths_path}/crosswalks.pt'
    lights_weight=f'{weigths_path}/lights.pt'
    persons_cars_weigth=f'{weigths_path}/persons_cars.pt'
    
    can=True
    
    crosswalk_model=YOLO(crosswalk_weight)
    lights_model=YOLO(lights_weight)
    persons_cars_model=YOLO(persons_cars_weigth)
    
    crosswalks_results = crosswalk_model(img_path,conf=conf)  # results list
    
    if len(crosswalks_results[0].boxes.xyxy.tolist())<1:
        return False
    
    lights_results=lights_model(img_path,conf=conf)
    
    if len(lights_results[0].boxes.xyxy.tolist())!=1:
        return False
    
    dic_type_lights=lights_results[0].names
    box=lights_results[0].boxes
    val=int(box.cls.tolist()[0])
    if dic_type_lights[val]!='verde':
        return False
    
    return True