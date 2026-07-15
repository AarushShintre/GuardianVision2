import argparse
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "SPHAR-Dataset"
VIDEOS_DIR = DATASET_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "output_frames"


def calculate_center(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))


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


def process_video_to_person_data(video_path, output_dir=OUTPUT_DIR):
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    tracked_boxes = {}
    next_box_id = 0

    print("Processing video and detecting people...")
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)

        for box in boxes:
            x, y, w, h = box
            matched = False
            center = calculate_center(x, y, w, h)

            for box_id, box_data in tracked_boxes.items():
                if calculate_iou(box, box_data["last_box"]) > 0.5:
                    crop = frame[y:y + h, x:x + w]
                    crop_filename = output_dir / f"person_{box_id}_frame_{frame_number:04d}.jpg"
                    cv2.imwrite(str(crop_filename), crop)
                    tracked_boxes[box_id]["images"].append(str(crop_filename))
                    tracked_boxes[box_id]["centers"].append(center)
                    tracked_boxes[box_id]["last_box"] = box
                    matched = True
                    break

            if not matched:
                crop = frame[y:y + h, x:x + w]
                crop_filename = output_dir / f"person_{next_box_id}_frame_{frame_number:04d}.jpg"
                cv2.imwrite(str(crop_filename), crop)
                tracked_boxes[next_box_id] = {
                    "images": [str(crop_filename)],
                    "centers": [center],
                    "last_box": box,
                }
                next_box_id += 1

        frame_number += 1

    cap.release()
    print("Video processing complete.")
    return tracked_boxes


def convert_to_desired_format(tracked_boxes):
    person_data = {}
    for person_id, box_data in tracked_boxes.items():
        person_data[person_id] = [box_data["images"], box_data["centers"]]
    return person_data


def parse_args():
    parser = argparse.ArgumentParser(description="Extract person crops and centers from a video.")
    parser.add_argument(
        "input_video",
        nargs="?",
        default=None,
        help="Path to the input video. If omitted, the first SPHAR MP4 under SPHAR-Dataset/videos is used.",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for extracted person crops.")
    return parser.parse_args()


def find_default_video(videos_dir=VIDEOS_DIR):
    videos_dir = Path(videos_dir)
    if not videos_dir.exists():
        raise FileNotFoundError(
            f"SPHAR videos directory not found: {videos_dir}. "
            "Pass an input video path or place videos under SPHAR-Dataset/videos."
        )

    try:
        return next(videos_dir.rglob("*.mp4"))
    except StopIteration as error:
        raise FileNotFoundError(
            f"No MP4 videos found under: {videos_dir}. "
            "Pass an input video path or populate SPHAR-Dataset/videos."
        ) from error


def main():
    args = parse_args()
    try:
        video_path = Path(args.input_video) if args.input_video else find_default_video()
        tracked_boxes = process_video_to_person_data(video_path, args.output_dir)
    except Exception as error:
        print(error)
        return 1

    person_data = convert_to_desired_format(tracked_boxes)
    print("\nGenerated Person Data:")
    for person_id, data in person_data.items():
        print(f"\nPerson {person_id}:")
        print(f"Number of frames: {len(data[0])}")
        print(f"Centers: {data[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
