import os
import random
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from constants import BEHAVIORS

# Directory for temporary storage of extracted frames
TEMP_FRAMES_DIR = "C:/Users/gangliagurdian/GuardianVision2/SPHAR-Dataset/temp_frames"
os.makedirs(TEMP_FRAMES_DIR, exist_ok=True)

def extract_frames_from_video(video_path, behavior):
    # ... (keep the existing implementation)

def get_training_videos():
    # ... (keep the existing implementation)

class SPHARDataset(Dataset):
    # ... (keep the existing implementation)

class SimpleCNN(nn.Module):
    # ... (keep the existing implementation)

def train_model():
    # Extract frames
    get_training_videos()

    # Set up dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = SPHARDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print("[DEBUG] Dataloader initialized.")

    # Model Initialization
    model = SimpleCNN(num_classes=len(BEHAVIORS))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("[DEBUG] Model initialized.")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[DEBUG] Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss/len(dataloader):.4f}")

    # Save the model
    MODEL_SAVE_PATH = "model.pth"
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[DEBUG] Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()