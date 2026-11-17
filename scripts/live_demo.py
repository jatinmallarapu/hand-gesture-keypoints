"""Show a smoothed prediction and the observed hand skeleton without saving video."""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from handgest.extract import detect, make_landmarker
from handgest.predict import load_predictor

BONES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="models/hand_landmarker.task")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    predictor = load_predictor(args.checkpoint, args.threshold)
    frames, history = deque(maxlen=32), deque(maxlen=3)
    capture = cv2.VideoCapture(args.camera)
    label, confidence = "no_gesture", 0.0
    start = time.monotonic()
    last = -1
    try:
        if not capture.isOpened():
            raise RuntimeError("Camera not available")
        with make_landmarker(args.model) as landmarker:
            i = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = max(last + 1, int((time.monotonic() - start) * 1000))
                last = timestamp
                points, found = detect(landmarker, frame, timestamp)
                frames.append(points)
                if found:
                    y, x = frame.shape[:2]
                    pixels = np.rint(points[:, :2] * [x, y]).astype(int)
                    for a, b in BONES:
                        cv2.line(frame, tuple(pixels[a]), tuple(pixels[b]), (90, 255, 150), 2)
                if len(frames) == 32 and i % 4 == 0:
                    result = predictor.predict(np.stack(frames))
                    history.append(result["label"])
                    if len(history) == 3 and len(set(history)) == 1:
                        label, confidence = result["label"], result["confidence"]
                cv2.putText(
                    frame,
                    f"{label} {confidence:.0%}  |  Q: quit",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (90, 255, 150),
                    2,
                )
                cv2.imshow("Hand Gesture Keypoints", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                i += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
