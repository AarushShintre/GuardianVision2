import numpy as np
import cv2
import glob
import csv
from run_model import predict

# Define behaviors and their color mappings
BEHAVIORS = ['falling', 'hitting', 'igniting', 'kicking', 'luggage', 'murdering', 
            'neutral', 'panicking', 'running', 'sitting', 'stealing', 'vandalizing', 'walking']

SAFE_BEHAVIORS = ['neutral', 'walking', 'sitting']

def get_box_color(behavior):
    """Return BGR color tuple based on behavior"""
    return (0, 255, 0) if behavior in SAFE_BEHAVIORS else (0, 0, 255)

def process_video_with_behaviors(behavior_list):
    video_path = r"C:\code\hackathon24\SPHAR-Dataset\videos\hitting\uccrime_Fighting049_x264.mp4"
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
            break

        boxes, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
        display_frame = frame.copy()

        # Get current behavior
        current_behavior = predict(frame) # call the predict
        box_color = get_box_color(current_behavior)

        for box in boxes:
            x, y, w, h = box
            matched = False
            center = (int(x + w/2), int(y + h/2))
            
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
            cv2.putText(display_frame, f'Person {next_box_id-1}', (x, y-25),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
            cv2.putText(display_frame, current_behavior, (x, y-5),
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

def main():
    # Example behavior list - you would need to provide the actual behavior list
    # This is just a sample - replace with your actual behavior data
    behavior_list = ['neutral'] * 50 + ['running'] * 30 + ['hitting'] * 40 + ['walking'] * 30
    
    # Process video with behavior analysis
    process_video_with_behaviors(behavior_list)

if __name__ == "__main__":
    main()