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

# Function to extract frames from videos shorter than 30 seconds
def extract_frames_from_video(video_path, behavior):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success = True

    # Check video duration
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    if duration > 30 :
        print(f"[DEBUG] Skipping {video_path}, duration exceeds 30 seconds.")
        return

    frames_dir = os.path.join(TEMP_FRAMES_DIR, behavior)
    os.makedirs(frames_dir, exist_ok=True)

    while success:
        success, frame = cap.read()
        if success:
            frame_path = os.path.join(frames_dir, f"{os.path.basename(video_path)}_frame_{frame_count}.jpeg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1

    cap.release()
    print(f"[DEBUG] Extracted {frame_count} frames from {video_path}")

def get_training_videos():
    for root, dirs, files in os.walk("C:/Users/gangliagurdian/GuardianVision2/SPHAR-Dataset/videos"):
        for dir_name in dirs:
            if dir_name in BEHAVIORS:
                action_type = dir_name
                dir_path = os.path.join(root, dir_name)
                all_videos = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.mp4')]
                sampled_videos = random.sample(all_videos, min(len(all_videos), 3))
                
                for video_path in sampled_videos:
                    extract_frames_from_video(video_path, action_type)

    print("[DEBUG] Video files processed and frames extracted.")

# Custom Dataset
class SPHARDataset(Dataset):
    def __init__(self, transform=None, sample_fraction=0.05):
        self.samples = []
        self.transform = transform

        # Load all the samples
        for behavior in BEHAVIORS:
            behavior_dir = os.path.join(TEMP_FRAMES_DIR, behavior)
            if os.path.isdir(behavior_dir):
                for file_name in os.listdir(behavior_dir):
                    if file_name.endswith('.jpeg'):
                        file_path = os.path.join(behavior_dir, file_name)
                        label = BEHAVIORS.index(behavior)
                        self.samples.append((file_path, label))

        print(f"[DEBUG] Loaded {len(self.samples)} samples into the dataset.")

        # Randomly sample 10% of the dataset
        sample_size = max(1, int(len(self.samples) * sample_fraction))
        self.samples = random.sample(self.samples, sample_size)
        print(f"[DEBUG] Using {len(self.samples)} samples (10% of total) for training.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Extract frames
get_training_videos()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = SPHARDataset(transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
print("[DEBUG] Dataloader initialized.")

# Simplified CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Model Initialization
model = SimpleCNN(num_classes=len(BEHAVIORS))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("[DEBUG] Model initialized.")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 5
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
torch.sa(model.state_dict(), MODEL_SAVE_PATH)
print(f"[DEBUG] Model saved to {MODEL_SAVE_PATH}")
