"""Create synthetic landmark trajectories to exercise the pipeline without camera data."""

import argparse
from pathlib import Path

import numpy as np

from handgest import LABELS


def generate(root: Path, seed: int, repeats: int):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for subject in range(3):
        for label_id, label in enumerate(LABELS):
            for repeat in range(repeats):
                frames = int(rng.integers(24, 49))
                phase = np.linspace(0, 1, frames)
                geometry = np.zeros((21, 3), dtype=np.float32)
                geometry[:, 0] = np.linspace(-0.07, 0.07, 21)
                geometry[:, 1] = np.sin(np.linspace(0, np.pi, 21)) * 0.12
                for tip, spread in ((4, 0.04), (8, 0.06), (12, 0.08), (16, 0.06), (20, 0.05)):
                    geometry[tip, 1] += spread
                wrist = np.tile([0.42 + subject * 0.06, 0.50, 0.0], (frames, 1))
                if label_id < 4:
                    wrist[:, label_id // 2] += (-1 if label_id % 2 == 0 else 1) * 0.18 * phase
                elif label == "pinch":
                    geometry[4] = geometry[8] + [0.012, 0.004, 0]
                elif label == "thumbs_up":
                    geometry[4] += [-0.08, -0.14, 0]
                    geometry[[8, 12, 16, 20], 1] *= 0.45
                points = wrist[:, None] + geometry[None] * (0.8 + subject * 0.12)
                points += rng.normal(0, 0.0018, size=points.shape)
                mask = rng.random(frames) > 0.08
                if label == "no_gesture":
                    mask[:] = False
                points[~mask] = np.nan
                np.savez_compressed(
                    root / f"synthetic-{subject}-{label}-{repeat}.npz",
                    keypoints=points.astype(np.float32),
                    mask=mask,
                    fps=30.0,
                    label=label,
                    subject=f"synthetic-{subject}",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/synthetic"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=6, choices=range(2, 31))
    args = parser.parse_args()
    generate(args.out, args.seed, args.repeats)
