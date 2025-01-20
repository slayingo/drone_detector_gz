import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# YOLO
import torch
print(torch.cuda.is_available())  # True döndürmelidir
import sympy
print(sympy.__file__)
import sys
print(sys.path)

# Etiketlerin bulunduğu klasör yolu


from ultralytics import YOLO

import pandas as pd

dataset = r'C:\Notlarim\Python_Projelerim\Projelerim\drone_CNN\yolo\dataset'

#training
!yolo task = detect mode=train model= yolov8m.pt data={dataset}/data.yaml epochs = 1000 imgsz = 640

#val 
!yolo task=detect mode=val model="C:/Notlarim/Python_Projelerim/Projelerim/Yolo/runs/detect/train/weights/best.pt" data="C:/Notlarim/Python_Projelerim/Projelerim/Yolo/dataset/data.yaml"

#predict
!yolo task = detect mode = predict model="C:/Notlarim/Python_Projelerim/Projelerim/drone_CNN/yolo/runs/detect/train4/weights/best.pt" data=dataset

#own test
!yolo task=detect mode=predict model="C:/Notlarim/Python_Projelerim/Projelerim/drone_CNN/yolo/runs/detect/train4/weights/best.pt" source=r'C:\Notlarim\Python_Projelerim\Projelerim\drone_CNN\yolo\dataset\test\images'


df_result = pd.read_csv('C:\\Notlarim\\Python_Projelerim\\Projelerim\\drone_CNN\\yolo\\runs\\detect\\train8\\results.csv')
last_epoch = df_result.iloc[-1]


precision = last_epoch['metrics/precision(B)']
recall = last_epoch['metrics/recall(B)']

f1_score = 2 * (precision * recall) / (precision + recall)

# F1 score
print(f"F1 Skoru: {f1_score:.4f}")


