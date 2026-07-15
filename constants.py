from pathlib import Path

from config_loader import load_app_config

BEHAVIORS =  ['falling', 'hitting', 'igniting', 'kicking', 'luggage', 'murdering', 'neutral', 'panicking', 'running', 'sitting', 'stealing', 'vandalizing', 'walking']

BASE_DIR = Path(__file__).resolve().parent
APP_CONFIG = load_app_config()
MODEL_SAVE_PATH = BASE_DIR / "behavior_model_final.pth"
INFERENCE_MODEL_PATH = APP_CONFIG["inference_model"]
DEFAULT_OUTPUT_VIDEO = APP_CONFIG["default_output_video"]
DEFAULT_TRACKING_CSV = APP_CONFIG["default_tracking_csv"]
FPS_FALLBACK = APP_CONFIG["fps_fallback"]
SAFE_BEHAVIORS = set(APP_CONFIG["safe_behaviors"])
DETECTION_METHOD = APP_CONFIG["detection_method"]
HOG_HIT_THRESHOLD = APP_CONFIG["hog_hit_threshold"]
HOG_WIN_STRIDE = APP_CONFIG["hog_win_stride"]
HOG_PADDING = APP_CONFIG["hog_padding"]
HOG_SCALE = APP_CONFIG["hog_scale"]
YOLO_CFG = APP_CONFIG["yolo_cfg"]
YOLO_WEIGHTS = APP_CONFIG["yolo_weights"]
YOLO_NAMES = APP_CONFIG["yolo_names"]
YOLO_CONFIDENCE_THRESHOLD = APP_CONFIG["yolo_confidence_threshold"]
YOLO_NMS_THRESHOLD = APP_CONFIG["yolo_nms_threshold"]
YOLO_PERSON_CLASS = APP_CONFIG["yolo_person_class"]
TRACKING_ENABLED = APP_CONFIG["tracking_enabled"]
TRACKING_IOU_THRESHOLD = APP_CONFIG["tracking_iou_threshold"]
TRACKING_MAX_MISSING_FRAMES = APP_CONFIG["tracking_max_missing_frames"]
