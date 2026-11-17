"""Record local, consented clips with a visual countdown and collision-free names."""

import argparse
import re
import time
from pathlib import Path

import cv2

from handgest import LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gesture", choices=LABELS, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9-]{1,32}", args.subject):
        parser.error("Use a pseudonymous ID with letters, digits or hyphens")
    root = Path("data/raw") / args.gesture
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    capture = cv2.VideoCapture(args.camera)
    writer = None
    countdown = None
    target = None
    recording_end = None
    records = 0
    size = None
    try:
        if not capture.isOpened():
            raise RuntimeError("Camera not available")
        capture.set(cv2.CAP_PROP_FPS, 30)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            now = time.monotonic()
            if countdown is not None and now >= countdown:
                while True:
                    target = root / f"{args.subject}_{args.gesture}_{index:04d}.mp4"
                    index += 1
                    if not target.exists():
                        break
                size = (frame.shape[1], frame.shape[0])
                writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
                if not writer.isOpened():
                    raise RuntimeError("Could not open MP4 writer")
                countdown = None
                recording_end = now + 2
                records = 0
            if writer is not None:
                if now < recording_end:
                    writer.write(
                        frame
                        if size == (frame.shape[1], frame.shape[0])
                        else cv2.resize(frame, size)
                    )
                    records += 1
                else:
                    writer.release()
                    writer = None
                    if records < 8:
                        target.unlink(missing_ok=True)
                        print("Discarded clip with too few frames")
                    else:
                        print(f"Saved {target} ({records} frames)")
            caption = "SPACE: record | Q: quit"
            if countdown is not None:
                caption = f"Recording in {max(1, int(countdown - now + 1))}"
            elif writer is not None:
                caption = "RECORDING"
            cv2.putText(frame, caption, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 60), 2)
            cv2.imshow("Hand Gesture Recorder", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == 32 and writer is None and countdown is None:
                countdown = time.monotonic() + 3
    finally:
        if writer is not None:
            writer.release()
            # Partial clips do not enter training after a camera disconnect or early exit.
            if target:
                target.unlink(missing_ok=True)
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
