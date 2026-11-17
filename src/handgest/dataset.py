"""Load portable NumPy records and isolate people across train, validation and test."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import LABELS
from .features import augment, to_features


@dataclass(frozen=True)
class Clip:
    points: np.ndarray
    mask: np.ndarray
    label: str
    subject: str
    fps: float = 30.0


def load(root: Path) -> list[Clip]:
    clips = []
    for name in sorted(root.glob("*.npz")):
        with np.load(name, allow_pickle=False) as blob:
            clip = Clip(
                blob["keypoints"],
                blob["mask"].astype(bool),
                str(blob["label"].item()),
                str(blob["subject"].item()),
                float(blob["fps"].item()),
            )
        if (
            clip.label not in LABELS
            or not clip.subject
            or not np.isfinite(clip.fps)
            or clip.fps <= 0
        ):
            raise ValueError(f"Invalid metadata in {name.name}")
        to_features(clip.points, clip.mask)
        clips.append(clip)
    if not clips:
        raise ValueError("No processed clips found. Run extraction or the synthetic smoke fixture.")
    return clips


def split(clips: list[Clip], test: str, validation: str):
    subjects = {clip.subject for clip in clips}
    if test == validation or not {test, validation} <= subjects or len(subjects) < 3:
        raise ValueError("Use distinct validation and test subjects from at least three people")
    return (
        [x for x in clips if x.subject not in {test, validation}],
        [x for x in clips if x.subject == validation],
        [x for x in clips if x.subject == test],
    )


def folds(clips: list[Clip]):
    subjects = sorted({x.subject for x in clips})
    for index, test in enumerate(subjects):
        validation = subjects[(index + 1) % len(subjects)]
        yield test, validation, split(clips, test, validation)


def vectorize(clips: list[Clip], rng=None):
    arrays, labels = [], []
    for clip in clips:
        points, mask, label = clip.points, clip.mask, clip.label
        if rng is not None:
            points, mask, label = augment(points, mask, label, rng)
        arrays.append(to_features(points, mask))
        labels.append(LABELS.index(label))
    return np.stack(arrays), np.asarray(labels, dtype=np.int64)


class GestureDataset:
    """PyTorch DataLoader can use this map-style dataset without importing torch here."""

    def __init__(self, clips: list[Clip], train: bool = False, seed: int = 0):
        self.clips = clips
        self.rng = np.random.default_rng(seed) if train else None

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        features, labels = vectorize([self.clips[index]], self.rng)
        return features[0], labels[0]
