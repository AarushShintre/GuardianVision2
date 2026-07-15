def calculate_iou(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    box_a_area = max(0, aw) * max(0, ah)
    box_b_area = max(0, bw) * max(0, bh)
    union_area = box_a_area + box_b_area - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_missing_frames=10):
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self.next_track_id = 0
        self.tracks = {}

    def update(self, detections):
        assignments = []
        matched_track_ids = set()

        for detection in detections:
            detection_box = self._box_from_detection(detection)
            best_track_id = None
            best_iou = 0.0

            for track_id, track in self.tracks.items():
                if track_id in matched_track_ids:
                    continue
                iou = calculate_iou(detection_box, track["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None and best_iou >= self.iou_threshold:
                track_id = best_track_id
            else:
                track_id = self.next_track_id
                self.next_track_id += 1

            self.tracks[track_id] = {"box": detection_box, "missing_frames": 0}
            matched_track_ids.add(track_id)
            assignments.append(track_id)

        for track_id in list(self.tracks):
            if track_id not in matched_track_ids:
                self.tracks[track_id]["missing_frames"] += 1
                if self.tracks[track_id]["missing_frames"] > self.max_missing_frames:
                    del self.tracks[track_id]

        return assignments

    def _box_from_detection(self, detection):
        return (
            int(detection["x"]),
            int(detection["y"]),
            int(detection["width"]),
            int(detection["height"]),
        )
