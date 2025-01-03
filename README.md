# GuardianVision2

GuardianVision2 is an advanced surveillance system developed during a collaborative hackathon. It detects and analyzes human behaviors in real-time video feeds, utilizing supervised learning techniques and YOLOv3 for object detection to identify specific actions and behaviors, thereby enabling proactive security measures.

## Features

- **Real-Time Behavior Detection:** Analyzes live video streams to identify predefined behaviors.
- **Model Training and Prediction:** Facilitates the training of machine learning models using supervised learning techniques and the prediction of behaviors.
- **Data Handling:** Manages data processing tasks and stores tracking information for individuals detected in the video footage.
- **Configuration Management:** Allows for flexible adjustments to the system's parameters through configuration files.

## Technologies Used

- **Programming Language:** Python
- **Machine Learning Framework:** PyTorch
- **Computer Vision Library:** OpenCV
- **Object Detection Model:** YOLOv3 (You Only Look Once, Version 3)
- **Data Handling:** CSV for tracking data
- **Configuration Management:** TOML for configuration files

## Dataset

The project utilizes the SPHAR (Surveillance Person and Human Activity Recognition) dataset, which contains annotated video footage of various human activities. This dataset is essential for training and evaluating the behavior detection models using supervised learning. The dataset is included in the repository under the `SPHAR-Dataset` directory.
