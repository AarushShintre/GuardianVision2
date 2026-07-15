from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError as error:
    tomllib = None
    TOML_IMPORT_ERROR = error
else:
    TOML_IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent
APP_CONFIG_PATH = BASE_DIR / "app_config.toml"

DEFAULT_CONFIG = {
    "inference_model": "model.pth",
    "default_output_video": "behavior_detection.mp4",
    "default_tracking_csv": "behavior_tracking.csv",
    "fps_fallback": 20,
    "safe_behaviors": ["neutral", "sitting", "walking"],
    "detection_method": "hog",
    "hog_hit_threshold": 0.0,
    "hog_win_stride": [8, 8],
    "hog_padding": [32, 32],
    "hog_scale": 1.05,
    "yolo_cfg": "yolov3.cfg",
    "yolo_weights": "yolov3.weights",
    "yolo_names": "coco.names",
    "yolo_confidence_threshold": 0.5,
    "yolo_nms_threshold": 0.4,
    "yolo_person_class": "person",
    "tracking_enabled": True,
    "tracking_iou_threshold": 0.3,
    "tracking_max_missing_frames": 10,
}


def resolve_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def load_app_config(config_path=APP_CONFIG_PATH):
    if tomllib is None:
        raise RuntimeError("TOML loading requires Python 3.11+ tomllib.") from TOML_IMPORT_ERROR

    config = DEFAULT_CONFIG.copy()
    config_path = Path(config_path)
    if config_path.exists():
        with config_path.open("rb") as config_file:
            loaded_config = tomllib.load(config_file)
        config.update(loaded_config.get("guardianvision", {}))

    return {
        "inference_model": resolve_path(config["inference_model"]),
        "default_output_video": resolve_path(config["default_output_video"]),
        "default_tracking_csv": resolve_path(config["default_tracking_csv"]),
        "fps_fallback": float(config["fps_fallback"]),
        "safe_behaviors": list(config["safe_behaviors"]),
        "detection_method": config["detection_method"],
        "hog_hit_threshold": float(config["hog_hit_threshold"]),
        "hog_win_stride": tuple(config["hog_win_stride"]),
        "hog_padding": tuple(config["hog_padding"]),
        "hog_scale": float(config["hog_scale"]),
        "yolo_cfg": resolve_path(config["yolo_cfg"]),
        "yolo_weights": resolve_path(config["yolo_weights"]),
        "yolo_names": resolve_path(config["yolo_names"]),
        "yolo_confidence_threshold": float(config["yolo_confidence_threshold"]),
        "yolo_nms_threshold": float(config["yolo_nms_threshold"]),
        "yolo_person_class": config["yolo_person_class"],
        "tracking_enabled": bool(config["tracking_enabled"]),
        "tracking_iou_threshold": float(config["tracking_iou_threshold"]),
        "tracking_max_missing_frames": int(config["tracking_max_missing_frames"]),
    }
