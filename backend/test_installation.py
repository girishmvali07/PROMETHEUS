import flask
import cv2
import torch
import ultralytics
import numpy

print("PROMETHEUS-X SYSTEM CHECK")
print("-------------------------")

print("Flask:", flask.__version__)
print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("NumPy:", numpy.__version__)

print("CUDA Available:", torch.cuda.is_available())

print("-------------------------")
print("ALL LIBRARIES IMPORTED SUCCESSFULLY!")