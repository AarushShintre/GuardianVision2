import numpy as np
import cv2
import glob
import os

def calculate_center(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

def process_video_to_person_data(video_path, output_dir="output_frames"):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

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

            # Calculate center for this detection
            center = calculate_center(x, y, w, h)

            for box_id, box_data in tracked_boxes.items():
                if calculate_iou(box, box_data['last_box']) > 0.5:
                    crop = frame[y:y + h, x:x + w]
                    crop_filename = f"{output_dir}/person_{box_id}_frame_{frame_number:04d}.jpg"
                    cv2.imwrite(crop_filename, crop)  # Save frame as image
                    tracked_boxes[box_id]['images'].append(crop_filename)
                    tracked_boxes[box_id]['centers'].append(center)
                    tracked_boxes[box_id]['last_box'] = box
                    matched = True
                    break

            if not matched:
                crop = frame[y:y + h, x:x + w]
                crop_filename = f"{output_dir}/person_{next_box_id}_frame_{frame_number:04d}.jpg"
                cv2.imwrite(crop_filename, crop)  # Save frame as image
                tracked_boxes[next_box_id] = {
                    'images': [crop_filename],
                    'centers': [center],
                    'last_box': box
                }
                next_box_id += 1

        frame_number += 1

    cap.release()
    print("Video processing complete.")
    return tracked_boxes

def convert_to_desired_format(tracked_boxes):
    """
    Convert tracked boxes dictionary into the desired output format.
    """
    person_data = {}
    for person_id, box_data in tracked_boxes.items():
        person_data[person_id] = [box_data['images'], box_data['centers']]
    return person_data

if __name__ == "__main__":
    # Input video path
    video_path = r"./SPHAR-Dataset/videos/kicking/bitint_kick_0001.mp4"
    
    # Process video and get person data
    tracked_boxes = process_video_to_person_data(video_path)

    # Convert to desired output format
    person_data = convert_to_desired_format(tracked_boxes)

    # Print the result
    print("\nGenerated Person Data:")
    for person_id, data in person_data.items():
        print(f"\nPerson {person_id}:")
        print(f"Number of frames: {len(data[0])}")
        print(f"Centers: {data[1]}")
