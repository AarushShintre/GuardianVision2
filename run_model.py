import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run GuardianVision video inference.")
    parser.add_argument("input_video", help="Path to the input MP4 video.")
    parser.add_argument("--output", default="run_model_output.mp4", help="Path for the processed output video.")
    parser.add_argument("--csv", default="run_model_tracking.csv", help="Path for the tracking CSV output.")
    return parser.parse_args()


def run_behavior_pipeline(input_video, output_path, csv_path):
    input_video = Path(input_video)
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    behavior_script = Path(__file__).resolve().parent / "behavior.py"
    command = [
        sys.executable,
        str(behavior_script),
        str(input_video),
        "--output",
        str(output_path),
        "--csv",
        str(csv_path),
    ]
    return subprocess.run(command, capture_output=True, text=True)


def main():
    args = parse_args()
    try:
        result = run_behavior_pipeline(args.input_video, args.output, args.csv)
    except Exception as error:
        print(f"Failed to run inference: {error}", file=sys.stderr)
        return 1

    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.returncode != 0:
        print(f"Behavior processing failed with exit code {result.returncode}.", file=sys.stderr)
        return result.returncode

    print(f"Run model output video: {args.output}")
    print(f"Run model tracking CSV: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
