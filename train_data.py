import numpy as np
import cv2

def preprocess_input(person_data):
    """
    Preprocess the input data for the model.

    Parameters:
    - person_data: Dictionary {1: [images, centers], 2: [images, centers], ...}

    Returns:
    - frame_input: Combined frame (numpy array).
    - centers_input: Flattened list of all person centers.
    - distances_input: Pairwise distances between centers.
    """
    all_centers = []
    combined_frame = None

    for person_id, (images, centers) in person_data.items():
        # Use the first image as the representative frame
        frame = cv2.imread(images[0])  # Read the first image for this person
        frame = cv2.resize(frame, (224, 224))  # Resize to model's expected input size

        if combined_frame is None:
            combined_frame = frame
        else:
            # Overlay the frames (ensure alignment before combining if needed)
            combined_frame = cv2.addWeighted(combined_frame, 0.5, frame, 0.5, 0)

        # Add the centers for this person
        all_centers.extend(centers)

    # Calculate pairwise distances between centers
    distances_input = calculate_pairwise_distances(all_centers)

    # Flatten the centers
    centers_input = np.array(all_centers).flatten()

    return combined_frame, centers_input, distances_input

def calculate_pairwise_distances(centers):
    """
    Calculate pairwise distances between centers.

    Parameters:
    - centers: List of center coordinates [(x1, y1), (x2, y2), ...]

    Returns:
    - A numpy array of pairwise distances
    """
    distances = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
            distances.append(dist)
    return np.array(distances)