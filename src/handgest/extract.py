"""Read one hand per frame without storing raw camera frames in output records."""

import argparse
import csv
from pathlib import Path

import numpy as np

from . import LABELS


def make_landmarker(path: str):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


def detect(landmarker, frame, timestamp):
    import cv2
    import mediapipe as mp

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect_for_video(image, timestamp)
    if result.hand_landmarks:
        points = np.array([(p.x, p.y, p.z) for p in result.hand_landmarks[0]], dtype=np.float32)
        return points, True
    return np.full((21, 3), np.nan, dtype=np.float32), False


def extract_video(path: Path, model: str, max_frames: int = 300):
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Video could not be opened")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and 1 <= fps <= 240 else 30.0
        points, flags = [], []
        # Each video gets a new VIDEO tracker with its own monotonic timestamp sequence.
        with make_landmarker(model) as landmarker:
            for index in range(max_frames + 1):
                ok, frame = capture.read()
                if not ok:
                    break
                if index == max_frames:
                    raise ValueError("Video exceeds frame limit")
                if frame.shape[0] * frame.shape[1] > 4_000_000:
                    raise ValueError("Video exceeds frame size limit")
                point, found = detect(landmarker, frame, int(index * 1000 / fps))
                points.append(point)
                flags.append(found)
        if not points:
            raise ValueError("No decodable video frames")
        return np.stack(points), np.asarray(flags, dtype=bool), fps
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", default="models/hand_landmarker.task")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    videos = sorted(args.raw.glob("*/*.mp4"))
    if not videos:
        parser.error("No raw MP4 clips found")
    with (args.out / "extraction.csv").open("w", encoding="utf-8", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["clip", "subject", "label", "detection_rate", "status"])
        for path in videos:
            label = path.parent.name
            if label not in LABELS:
                continue
            subject = path.stem.split("_", 1)[0]
            target = args.out / f"{label}--{path.stem}.npz"
            try:
                points, mask, fps = extract_video(path, args.model)
            except ValueError as error:
                target.unlink(missing_ok=True)
                writer.writerow([path.name, subject, label, "", str(error)])
                continue
            rate = float(mask.mean())
            # Empty scenes are valid negatives. Dropping them would change the class definition.
            keep = rate >= 0.5 or label == "no_gesture"
            writer.writerow(
                [path.name, subject, label, f"{rate:.4f}", "kept" if keep else "dropped"]
            )
            if keep:
                np.savez_compressed(
                    target, keypoints=points, mask=mask, fps=fps, label=label, subject=subject
                )
            else:
                target.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
