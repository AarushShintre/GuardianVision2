import argparse
import csv
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from constants import (
    APP_CONFIG,
    BEHAVIORS,
    DEFAULT_OUTPUT_VIDEO,
    DEFAULT_TRACKING_CSV,
    FPS_FALLBACK,
    INFERENCE_MODEL_PATH,
    SAFE_BEHAVIORS,
    TRACKING_ENABLED,
    TRACKING_IOU_THRESHOLD,
    TRACKING_MAX_MISSING_FRAMES,
)
from detectors import get_detector
from model import SimpleCNN
from tracker import SimpleTracker

def get_box_color(behavior):
    return (0, 255, 0) if behavior in SAFE_BEHAVIORS else (0, 0, 255)

def predict(image, model, device):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(pil_image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = BEHAVIORS[predicted.item()]
    return predicted_class, confidence.item()

def clamp_detection_to_frame(detection, frame_width, frame_height):
    x = max(0, int(detection["x"]))
    y = max(0, int(detection["y"]))
    width = int(detection["width"])
    height = int(detection["height"])
    x2 = min(frame_width, x + max(0, width))
    y2 = min(frame_height, y + max(0, height))
    return x, y, x2, y2

def process_video_with_behaviors(model, device, input_video_path, output_video_path, csv_output_path):
    input_video_path = Path(input_video_path)
    output_video_path = Path(output_video_path)
    csv_output_path = Path(csv_output_path)

    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")
    if input_video_path.resolve() == output_video_path.resolve():
        raise ValueError(f"Input and output video paths must be different: {input_video_path}")

    detector = get_detector(APP_CONFIG)
    tracker = None
    if TRACKING_ENABLED:
        tracker = SimpleTracker(
            iou_threshold=TRACKING_IOU_THRESHOLD,
            max_missing_frames=TRACKING_MAX_MISSING_FRAMES,
        )

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = FPS_FALLBACK
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    behavior_output = cv2.VideoWriter(
        str(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height)
    )
    if not behavior_output.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video for writing: {output_video_path}")

    frame_count = 0

    print("Processing video with behavior analysis...")
    with csv_output_path.open("w", newline="") as csv_file:
        fieldnames = [
            "frame_number",
            "timestamp_seconds",
            "detection_id",
            "track_id",
            "x",
            "y",
            "width",
            "height",
            "detector",
            "detector_confidence",
            "behavior",
            "behavior_confidence",
            "is_safe",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            track_ids = tracker.update(detections) if tracker else [None] * len(detections)
            display_frame = frame.copy()

            timestamp_seconds = frame_count / fps

            # frame_number is zero-based to match the first decoded frame from OpenCV.
            for detection_id, detection in enumerate(detections):
                track_id = track_ids[detection_id]
                x, y, x2, y2 = clamp_detection_to_frame(detection, frame_width, frame_height)
                w = x2 - x
                h = y2 - y
                if w <= 0 or h <= 0:
                    continue

                person_crop = frame[y:y2, x:x2]
                if person_crop.size == 0:
                    continue

                current_behavior, behavior_confidence = predict(person_crop, model, device)
                detector_confidence = float(detection.get("confidence", 0.0))
                box_color = get_box_color(current_behavior)
                is_safe = current_behavior in SAFE_BEHAVIORS
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)
                person_label = f'ID {track_id}' if track_id is not None else f'Person {detection_id}'
                cv2.putText(display_frame, person_label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
                cv2.putText(display_frame, f'{current_behavior} {behavior_confidence:.2f} | {detection["detector"]} {detector_confidence:.2f}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                csv_writer.writerow({
                    "frame_number": frame_count,
                    "timestamp_seconds": f"{timestamp_seconds:.3f}",
                    "detection_id": detection_id,
                    # When tracking is disabled, track_id is intentionally blank.
                    "track_id": "" if track_id is None else track_id,
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "detector": detection["detector"],
                    "detector_confidence": f"{detector_confidence:.6f}",
                    "behavior": current_behavior,
                    "behavior_confidence": f"{behavior_confidence:.6f}",
                    "is_safe": is_safe,
                })

            cv2.putText(display_frame, f'Frame: {frame_count} | Detections: {len(detections)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            behavior_output.write(display_frame)

            frame_count += 1

    cap.release()
    behavior_output.release()
    print(f"Processed video saved to: {output_video_path}")
    print(f"Tracking CSV saved to: {csv_output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Process a video with behavior detection overlays.")
    parser.add_argument("input_video", help="Path to the input MP4 video.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_VIDEO, help="Path for the processed output video.")
    parser.add_argument("--csv", default=DEFAULT_TRACKING_CSV, help="Path for the tracking CSV output.")
    return parser.parse_args()

def normalize_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    return checkpoint

def load_model_state(model, model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        checkpoint = normalize_checkpoint(checkpoint)
        model.load_state_dict(checkpoint)
    except Exception as error:
        raise RuntimeError(f"Failed to load model checkpoint from {model_path}: {error}") from error

    print(f"Loaded model from: {model_path}")

def main():
    args = parse_args()
    device = torch.device("cpu")
    model = SimpleCNN(num_classes=len(BEHAVIORS))
    load_model_state(model, INFERENCE_MODEL_PATH)
    model.to(device)
    model.eval()

    process_video_with_behaviors(model, device, args.input_video, args.output, args.csv)

if __name__ == "__main__":
    main()
