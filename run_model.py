import os
import random
import cv2  # OpenCV for video processing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from constants import BEHAVIORS  # Assuming BEHAVIORS is defined in constants.py

def predict(image_path):
    model.load
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        print(f"[DEBUG] Prediction made for image: {image_path}")
        return BEHAVIORS[predicted.item()]

# Example usage of the prediction function
test_image = "C:/code/hackathon24/SPHAR-Dataset/temp_frames/walking/example_frame_0.jpeg"
predicted_behavior = predict(test_image)
print(f"Predicted behavior: {predicted_behavior}")