import random
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from constants import BASE_DIR, BEHAVIORS
from model import SimpleCNN


DATASET_DIR = BASE_DIR / "SPHAR-Dataset"
VIDEOS_DIR = DATASET_DIR / "videos"
TEMP_FRAMES_DIR = DATASET_DIR / "temp_frames"
MODEL_SAVE_PATH = BASE_DIR / "model.pth"


def validate_dataset_path(videos_dir=VIDEOS_DIR):
    videos_dir = Path(videos_dir)
    if not videos_dir.exists():
        raise FileNotFoundError(
            f"SPHAR videos directory not found: {videos_dir}. "
            "Expected SPHAR dataset videos under SPHAR-Dataset/videos."
        )
    if not any(videos_dir.rglob("*.mp4")):
        raise FileNotFoundError(
            f"No MP4 training videos found under: {videos_dir}. "
            "The SPHAR dataset appears missing or empty."
        )


def extract_frames_from_video(video_path, behavior, temp_frames_dir=TEMP_FRAMES_DIR):
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    success = True

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        cap.release()
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    if duration > 30:
        cap.release()
        print(f"[DEBUG] Skipping {video_path}, duration exceeds 30 seconds.")
        return

    frames_dir = Path(temp_frames_dir) / behavior
    frames_dir.mkdir(parents=True, exist_ok=True)

    while success:
        success, frame = cap.read()
        if success:
            frame_path = frames_dir / f"{Path(video_path).name}_frame_{frame_count}.jpeg"
            cv2.imwrite(str(frame_path), frame)
            frame_count += 1

    cap.release()
    print(f"[DEBUG] Extracted {frame_count} frames from {video_path}")


def get_training_videos(videos_dir=VIDEOS_DIR, temp_frames_dir=TEMP_FRAMES_DIR, videos_per_behavior=3):
    validate_dataset_path(videos_dir)
    Path(temp_frames_dir).mkdir(parents=True, exist_ok=True)

    for behavior_dir in Path(videos_dir).rglob("*"):
        if not behavior_dir.is_dir() or behavior_dir.name not in BEHAVIORS:
            continue

        action_type = behavior_dir.name
        all_videos = list(behavior_dir.glob("*.mp4"))
        sampled_videos = random.sample(all_videos, min(len(all_videos), videos_per_behavior))

        for video_path in sampled_videos:
            extract_frames_from_video(video_path, action_type, temp_frames_dir)

    print("[DEBUG] Video files processed and frames extracted.")


class SPHARDataset(Dataset):
    def __init__(self, frames_dir=TEMP_FRAMES_DIR, transform=None, sample_fraction=0.05):
        self.samples = []
        self.transform = transform
        frames_dir = Path(frames_dir)

        for behavior in BEHAVIORS:
            behavior_dir = frames_dir / behavior
            if behavior_dir.is_dir():
                for image_path in behavior_dir.glob("*.jpeg"):
                    label = BEHAVIORS.index(behavior)
                    self.samples.append((image_path, label))

        print(f"[DEBUG] Loaded {len(self.samples)} samples into the dataset.")
        if not self.samples:
            raise FileNotFoundError(
                f"No extracted training frames found under: {frames_dir}. "
                "Run frame extraction with a valid SPHAR dataset before training."
            )

        sample_size = max(1, int(len(self.samples) * sample_fraction))
        self.samples = random.sample(self.samples, sample_size)
        print(f"[DEBUG] Using {len(self.samples)} samples ({sample_fraction:.0%} of total) for training.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def create_transform():
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_dataset(frames_dir=TEMP_FRAMES_DIR, sample_fraction=0.05):
    return SPHARDataset(
        frames_dir=frames_dir,
        transform=create_transform(),
        sample_fraction=sample_fraction,
    )


def create_dataloader(dataset, batch_size=8):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("[DEBUG] Dataloader initialized.")
    return dataloader


def train_model(dataloader, num_epochs=5, learning_rate=0.001, device=None):
    model = SimpleCNN(num_classes=len(BEHAVIORS))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("[DEBUG] Model initialized.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

    return model


def save_model(model, model_save_path=MODEL_SAVE_PATH):
    torch.save(model.state_dict(), model_save_path)
    print(f"[DEBUG] Model saved to {model_save_path}")


def main():
    get_training_videos()
    dataset = create_dataset()
    dataloader = create_dataloader(dataset)
    model = train_model(dataloader)
    save_model(model)


if __name__ == "__main__":
    main()
