from pathlib import Path

import cv2
import numpy as np


class HOGDetector:
    def __init__(self, win_stride, padding, scale, hit_threshold):
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self.hit_threshold = hit_threshold
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        boxes, weights = self.hog.detectMultiScale(
            frame,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
            hitThreshold=self.hit_threshold,
        )
        detections = []
        for box, weight in zip(boxes, weights):
            x, y, width, height = box
            detections.append({
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "confidence": float(weight),
                "detector": "hog",
            })
        return detections


class YOLOv3Detector:
    def __init__(
        self,
        cfg_path,
        weights_path,
        names_path,
        confidence_threshold,
        nms_threshold,
        person_class,
    ):
        self.cfg_path = Path(cfg_path)
        self.weights_path = Path(weights_path)
        self.names_path = Path(names_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.person_class = person_class
        self._validate_paths()
        self.class_names = self._load_class_names()
        if self.person_class not in self.class_names:
            raise ValueError(f"YOLO person class '{self.person_class}' was not found in {self.names_path}")
        self.person_class_id = self.class_names.index(self.person_class)
        self.net = cv2.dnn.readNetFromDarknet(str(self.cfg_path), str(self.weights_path))
        self.output_layer_names = self.net.getUnconnectedOutLayersNames()

    def _validate_paths(self):
        missing_paths = [
            path for path in [self.cfg_path, self.weights_path, self.names_path]
            if not path.exists()
        ]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"YOLOv3 detector requires missing file(s): {missing}")

    def _load_class_names(self):
        with self.names_path.open("r", encoding="utf-8") as names_file:
            return [line.strip() for line in names_file if line.strip()]

    def detect(self, frame):
        frame_height, frame_width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layer_names)

        boxes = []
        confidences = []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if class_id != self.person_class_id or confidence < self.confidence_threshold:
                    continue

                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, width, height])
                confidences.append(confidence)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        if len(indexes) == 0:
            return []

        detections = []
        for index in np.array(indexes).flatten():
            x, y, width, height = boxes[int(index)]
            detections.append({
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "confidence": float(confidences[int(index)]),
                "detector": "yolov3",
            })
        return detections


def get_detector(config):
    detection_method = config["detection_method"]
    if detection_method == "hog":
        return HOGDetector(
            win_stride=config["hog_win_stride"],
            padding=config["hog_padding"],
            scale=config["hog_scale"],
            hit_threshold=config["hog_hit_threshold"],
        )
    if detection_method == "yolov3":
        return YOLOv3Detector(
            cfg_path=config["yolo_cfg"],
            weights_path=config["yolo_weights"],
            names_path=config["yolo_names"],
            confidence_threshold=config["yolo_confidence_threshold"],
            nms_threshold=config["yolo_nms_threshold"],
            person_class=config["yolo_person_class"],
        )
    raise ValueError(f"Unsupported detection method: {detection_method}")
