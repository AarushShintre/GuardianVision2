import numpy as np
import cv2
import glob
import csv

def calculate_center(x, y, w, h):
    return (int(x + w/2), int(y + h/2))

def save_person_dict_to_csv(person_dict):
    # Define the CSV headers
    headers = ['person_id', 'frame_number', 'center_x', 'center_y']

    # Open CSV file for writing
    with open('person_tracking.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write headers
        writer.writerow(headers)

        # Write data for each person
        for person_id in person_dict:
            centers = person_dict[person_id][1]  # Get centers list

            # Write each frame's data
            for frame_num, center in enumerate(centers):
                if center:  # Only write if center exists
                    writer.writerow([
                        person_id,          # Person ID
                        frame_num,          # Frame number
                        center[0],          # Center X coordinate
                        center[1]           # Center Y coordinate
                    ])

    print(f"Data saved to person_tracking.csv")

def process_video():
    video_path = r"C:\code\hackathon24\SPHAR-Dataset\videos\kicking\bitint_kick_0001.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Get video dimensions for full video output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter for full video output
    full_output = cv2.VideoWriter(
        'full_detection.mp4',
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

    print("Processing video and detecting people...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)

        # Create a copy of the frame for drawing
        display_frame = frame.copy()

        for box in boxes:
            x, y, w, h = box
            matched = False

            # Calculate center for this detection
            center = calculate_center(x, y, w, h)

            for box_id, box_data in tracked_boxes.items():
                if calculate_iou(box, box_data['last_box']) > 0.5:
                    crop = frame[y:y+h, x:x+w]
                    tracked_boxes[box_id]['crops'].append(crop)
                    tracked_boxes[box_id]['centers'].append(center)
                    tracked_boxes[box_id]['last_box'] = box
                    matched = True
                    break

            if not matched:
                tracked_boxes[next_box_id] = {
                    'crops': [frame[y:y+h, x:x+w]],
                    'centers': [center],
                    'last_box': box
                }
                next_box_id += 1

            # Draw rectangle and center point
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(display_frame, center, 8, (255, 0, 0), -1)  # Large blue dot

            # Add person ID text
            for box_id, box_data in tracked_boxes.items():
                if np.array_equal(box, box_data['last_box']):
                    cv2.putText(display_frame, f'Person {box_id}', (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the frame with detections to the full video
        full_output.write(display_frame)

        cv2.imshow('Detections', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    full_output.release()

    # Create the final person dictionary and videos
    person_dict = {}

    for box_id, box_data in tracked_boxes.items():
        if len(box_data['crops']) > 10:  # Only process if enough frames
            out_path = f'person_{box_id}.mp4'
            h, w = box_data['crops'][0].shape[:2]

            out = cv2.VideoWriter(out_path, 
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps,
                                (w, h))

            for crop in box_data['crops']:
                crop = cv2.resize(crop, (w, h))
                out.write(crop)

            out.release()

            # Add to person dictionary
            person_dict[box_id] = [box_data['crops'], box_data['centers']]
            print(f'Person {box_id}: {len(box_data["crops"])} frames, {len(box_data["centers"])} center points')

    cv2.destroyAllWindows()
    return person_dict

def play_isolated_videos(person_dict):
    print("\nPlaying isolated videos with centers...")
    video_files = glob.glob('person_*.mp4')

    if not video_files:
        print("No isolated videos found!")
        return

    caps = [cv2.VideoCapture(file) for file in video_files]
    frame_count = {i: 0 for i in range(len(caps))}

    while True:
        all_done = True

        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret:
                all_done = False

                # Get center for this frame from person_dict
                if i in person_dict and frame_count[i] < len(person_dict[i][1]):
                    center = person_dict[i][1][frame_count[i]]
                    # Draw center point
                    cv2.circle(frame, center, 8, (255, 0, 0), -1)  # Large blue dot

                cv2.imshow(f'Person {i}', frame)
                frame_count[i] += 1

        if all_done or cv2.waitKey(30) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Process video and get person dictionary
    person_dict = process_video()

    # Print dictionary structure
    print("\nPerson Dictionary Structure:")
    for person_id in person_dict:
        print(f"\nPerson {person_id}:")
        print(f"Number of frames: {len(person_dict[person_id][0])}")
        print(f"Number of centers: {len(person_dict[person_id][1])}")
        print(f"Sample center coordinates: {person_dict[person_id][1][0]}")

    # Save tracking data to CSV
    save_person_dict_to_csv(person_dict)

    # Play isolated videos
    play_isolated_videos(person_dict) 
