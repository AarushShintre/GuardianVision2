import random
from pathlib import Path

from constants import BEHAVIORS


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "SPHAR-Dataset"
VIDEOS_DIR = DATASET_DIR / "videos"


def validate_videos_dir(videos_dir=VIDEOS_DIR):
    videos_dir = Path(videos_dir)
    if not videos_dir.exists():
        raise FileNotFoundError(
            f"SPHAR videos directory not found: {videos_dir}. "
            "Expected dataset videos under SPHAR-Dataset/videos."
        )
    if not any(videos_dir.rglob("*.mp4")):
        raise FileNotFoundError(
            f"No MP4 videos found under: {videos_dir}. "
            "The SPHAR dataset appears missing or empty."
        )


def get_training_videos(videos_dir=VIDEOS_DIR, sample_size=10):
    validate_videos_dir(videos_dir)
    videos = {}

    for behavior_dir in Path(videos_dir).rglob("*"):
        if not behavior_dir.is_dir() or behavior_dir.name not in BEHAVIORS:
            continue

        all_files = [path for path in behavior_dir.iterdir() if path.is_file()]
        sampled_files = random.sample(all_files, min(len(all_files), sample_size))
        videos[behavior_dir.name] = videos.get(behavior_dir.name, 0) + len(sampled_files)

    return videos


def main():
    try:
        videos = get_training_videos()
    except FileNotFoundError as error:
        print(error)
        return 1

    print(videos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
