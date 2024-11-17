import os
import random
from constants import BEHAVIORS  

def get_training_videos():
    # Initialize a dictionary to hold video counts for each behavior.
    videos = {}

    # Walk through the directory structure.
    for root, dirs, files in os.walk("../SPHAR-dataset/videos"):
        for dir_name in dirs:
            if dir_name in BEHAVIORS:
                actionType = dir_name  # Identify the action type based on the directory name.

                # Get the full path of the directory.
                dir_path = os.path.join(root, dir_name)

                # List all files in the directory.
                all_files = os.listdir(dir_path)

                # Sample up to 10 files (or fewer if less than 10 files exist).
                sampled_files = random.sample(all_files, min(len(all_files), 10))

                # Initialize the count for the action type if not already done.
                if actionType not in videos:
                    videos[actionType] = 0

                # Update the count with the number of sampled files.
                videos[actionType] += len(sampled_files)

    # return the resulting dictionary.
    return videos 
