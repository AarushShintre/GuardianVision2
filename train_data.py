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

# Directory for temporary storage of extracted frames
TEMP_FRAMES_DIR = "C:/code/hackathon24/SPHAR-Dataset/temp_frames"
os.makedirs(TEMP_FRAMES_DIR, exist_ok=True)

# Step 1: Extract frames from videos and save as .jpeg files
def extract_frames_from_video(video_path, behavior):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success = True

    frames_dir = os.path.join(TEMP_FRAMES_DIR, behavior)
    os.makedirs(frames_dir, exist_ok=True)

    while success:
        success, frame = cap.read()
        if success:
            frame_path = os.path.join(frames_dir, f"{os.path.basename(video_path)}_frame_{frame_count}.jpeg")
            cv2.imwrite(frame_path, frame)  # Save the frame as a .jpeg file
            frame_count += 1

    cap.release()
    print(f"[DEBUG] Extracted {frame_count} frames from {video_path}")

# Step 2: Process all videos and extract frames
def get_training_videos():
    for root, dirs, files in os.walk("C:/code/hackathon24/SPHAR-Dataset/videos"):
        for dir_name in dirs:
            if dir_name in BEHAVIORS:
                actionType = dir_name
                dir_path = os.path.join(root, dir_name)

                # Get all video files in the directory
                all_videos = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.mp4')]
                
                # Sample up to 3 videos (or fewer if less than 3 videos exist)
                sampled_videos = random.sample(all_videos, min(len(all_videos), 3))

                # Extract frames from the sampled videos
                for video_path in sampled_videos:
                    extract_frames_from_video(video_path, actionType)

    print(f"[DEBUG] Video files processed and frames extracted. Using {sampled_videos}")

# Step 3: Create a custom dataset using the extracted frames
class SPHARDataset(Dataset):
    def __init__(self, transform=None):
        self.samples = []
        self.transform = transform

        # Traverse the TEMP_FRAMES_DIR to get all .jpeg files
        for behavior in BEHAVIORS:
            behavior_dir = os.path.join(TEMP_FRAMES_DIR, behavior)
            if os.path.isdir(behavior_dir):
                for file_name in os.listdir(behavior_dir):
                    if file_name.endswith('.jpeg'):
                        file_path = os.path.join(behavior_dir, file_name)
                        label = BEHAVIORS.index(behavior)
                        self.samples.append((file_path, label))

        print(f"[DEBUG] Loaded {len(self.samples)} samples into the dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Step 4: Prepare the dataset and dataloader
get_training_videos()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = SPHARDataset(transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
print("[DEBUG] Dataloader initialized.")

# Step 5: Define the CNN model using a pretrained ResNet18
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(BEHAVIORS))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("[DEBUG] Model initialized and moved to device.")

# Step 6: Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("[DEBUG] Loss function and optimizer set up.")

# Step 7: Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"[DEBUG] Epoch [{epoch+1}/{num_epochs}] completed. Loss: {total_loss/len(dataloader):.4f}")

# Step 8: Save the trained model
MODEL_SAVE_PATH = "action_classification_model.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"[DEBUG] Model saved to {MODEL_SAVE_PATH}")

