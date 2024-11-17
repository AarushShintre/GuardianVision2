import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from constants import BEHAVIORS, SAFE_BEHAVIORS, MODEL_SAVE_PATH
from train_data import SimpleCNN

def get_box_color(behavior):
    return (0, 255, 0) if behavior in SAFE_BEHAVIORS else (0, 0, 255)

def predict(frame, model, device):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(pil_image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = BEHAVIORS[predicted.item()]
    return predicted_class

def process_video_with_behaviors(model, device):
    video_path = r"C:/Users/gangliagurdian/GuardianVision2/SPHAR-Dataset/videos/hitting/uccrime_Fighting049_x264.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    behavior_output = cv2.VideoWriter(
        'behavior_detection.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height)
    )

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    tracked_boxes = {}
    next_box_id = 0
    frame_count = 0

    print("Processing video with behavior analysis...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
        display_frame = frame.copy()

        current_behavior = predict(frame, model, device)
        box_color = get_box_color(current_behavior)

        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(display_frame, f'Person {next_box_id}', (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
            cv2.putText(display_frame, current_behavior, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            next_box_id += 1

        cv2.putText(display_frame, f'Frame: {frame_count} | Behavior: {current_behavior}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        behavior_output.write(display_frame)
        cv2.imshow('Behavior Detection', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    behavior_output.release()
    cv2.destroyAllWindows()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(BEHAVIORS))
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    process_video_with_behaviors(model, device)

if __name__ == "__main__":
    main()