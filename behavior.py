import os
import random
import cv2  # OpenCV for video processing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from constants import BEHAVIORS  # Assuming BEHAVIORS is defined in constants.py

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
    if duration > 30:
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
                sampled_videos = random.sample(all_videos, min(len(all_videos), 1))

                for video_path in sampled_videos:
                    extract_frames_from_video(video_path, action_type)

    print("[DEBUG] Video files processed and frames extracted.")

# Custom Dataset
class SPHARDataset(Dataset):
    def __init__(self, transform=None, sample_fraction=0.005):
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

        # Randomly sample the dataset based on fraction
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
        
        # Calculate the size of the output from conv layers
        self.conv_output_size = self._get_conv_output((3, 64, 64))
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def _get_conv_output(self, shape):
        input = torch.rand(1, *shape)
        output = self.conv_layers(input)
        return int(torch.prod(torch.tensor(output.size()[1:])))

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

# Save the model (optional, in case we want to save)
MODEL_SAVE_PATH = "model.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"[DEBUG] Model saved to {MODEL_SAVE_PATH}")

# Define behaviors and their color mappings
BEHAVIORS = ['falling', 'hitting', 'igniting', 'kicking', 'luggage', 'murdering', 
            'neutral', 'panicking', 'running', 'sitting', 'stealing', 'vandalizing', 'walking']

SAFE_BEHAVIORS = ['neutral', 'walking', 'sitting']

def get_box_color(behavior):
    """Return BGR color tuple based on behavior"""
    return (0, 255, 0) if behavior in SAFE_BEHAVIORS else (0, 0, 255)

def process_video_with_behaviors():
    video_path = r"C:/Users/gangliagurdian/GuardianVision2/SPHAR-Dataset/videos/hitting/uccrime_Fighting049_x264.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Get video dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    behavior_output = cv2.VideoWriter(
        'behavior_detection.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    # Initialize HOG detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Dictionary to store tracked boxes and their data
    tracked_boxes = {}
    next_box_id = 0

    def calculate_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0

    frame_count = 0
    print("Processing video with behavior analysis...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        boxes, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
        display_frame = frame.copy()

        # Get current behavior
        current_behavior = predict(frame)  # Use frame here
        box_color = get_box_color(current_behavior)

        for box in boxes:
            x, y, w, h = box
            matched = False
            center = (int(x + w / 2), int(y + h / 2))
            
            for box_id, box_data in tracked_boxes.items():
                if calculate_iou(box, box_data['last_box']) > 0.5:
                    tracked_boxes[box_id]['last_box'] = box
                    matched = True
                    break
            
            if not matched:
                tracked_boxes[next_box_id] = {
                    'last_box': box
                }
                next_box_id += 1

            # Draw rectangle with behavior-based color
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.circle(display_frame, center, 8, box_color, -1)
            
            # Add person ID and current behavior text
            cv2.putText(display_frame, f'Person {next_box_id - 1}', (x, y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
            cv2.putText(display_frame, current_behavior, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # Add frame counter and behavior to top-left corner
        cv2.putText(display_frame, f'Frame: {frame_count} | Behavior: {current_behavior}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write the frame with detections
        behavior_output.write(display_frame)
        
        # Display the frame
        cv2.imshow('Behavior Detection', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    behavior_output.release()
    cv2.destroyAllWindows()

def predict(frame):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = transform(pil_image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = BEHAVIORS[predicted.item()]
        print(f"Predicted class: {predicted_class}")  # Debug print
    return predicted_class

def main():
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    process_video_with_behaviors()

if __name__ == "__main__":
    main()