import streamlit as st
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Preprocessing Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Model Definition
class SimpleActionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SimpleActionClassifier, self).__init__()
        base_model = models.resnet18(pretrained=True)
        base_model.fc = nn.sigmoid(base_model.fc.in_features, num_classes)
        self.model = base_model

    def forward(self, x):
        return self.model(x)

# Prepare dataset
dataset = SemiSupervisedDataset(annotated_data, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model
model = SimpleActionClassifier(num_classes=len(set(action.split("_")[0] for action in annotated_data.keys())))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(3):  # Small epoch for demonstration
    for images, _, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                st.write(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

            st.success("Model trained successfully!")