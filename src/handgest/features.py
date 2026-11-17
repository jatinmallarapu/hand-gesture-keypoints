"""Separate palm-relative shape from wrist movement before adding velocities."""

import numpy as np

STEPS = 32
WIDTH = 132


def fill_gaps(points, mask):
    points = np.asarray(points, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if points.ndim != 3 or points.shape[1:] != (21, 3) or not len(points):
        raise ValueError("Expected at least one frame with shape (21, 3)")
    if mask.shape != (len(points),):
        raise ValueError("Mask must contain one value per frame")
    good = mask & np.isfinite(points).all(axis=(1, 2))
    if not good.any():
        return np.zeros_like(points)
    indices = np.arange(len(points))
    flat = points.reshape(len(points), -1)
    filled = np.column_stack([np.interp(indices, indices[good], x[good]) for x in flat.T])
    return filled.reshape(points.shape).astype(np.float32)


def resample(x, steps=STEPS):
    source = np.linspace(0, 1, len(x))
    target = np.linspace(0, 1, steps)
    return np.column_stack([np.interp(target, source, col) for col in x.T])


def to_features(points, mask, steps=STEPS):
    points = fill_gaps(points, mask)
    wrist = points[:, 0]
    delta = points - wrist[:, None]
    palm = float(np.median(np.linalg.norm(delta[:, 9, :2], axis=1)))
    if palm < 1e-5:
        return np.zeros((steps, WIDTH), dtype=np.float32)
    shape = (delta / palm).reshape(len(points), 63)
    trajectory = (wrist - wrist[0]) / palm
    signal = resample(np.column_stack((shape, trajectory)), steps)
    return np.column_stack((signal, np.diff(signal, axis=0, prepend=signal[:1]))).astype(np.float32)


def mirror(points, label):
    result = points.copy()
    result[..., 0] = 1 - result[..., 0]
    swapped = {"swipe_left": "swipe_right", "swipe_right": "swipe_left"}
    return result, swapped.get(label, label)


def augment(points, mask, label, rng):
    points = points.copy()
    mask = mask.copy()
    if rng.random() < 0.5:
        points, label = mirror(points, label)
    width = max(1, int(len(points) * rng.uniform(0.85, 1.0)))
    first = int(rng.integers(len(points) - width + 1))
    points, mask = points[first : first + width], mask[first : first + width]
    wrist = points[:, :1].copy()
    angle = np.deg2rad(rng.uniform(-15, 15))
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    points[..., :2] = (points[..., :2] - wrist[..., :2]) @ rotation.T * rng.uniform(
        0.9, 1.1
    ) + wrist[..., :2]
    points += rng.normal(0, 0.01, size=points.shape).astype(np.float32)
    valid = np.flatnonzero(mask)
    count = min(int(len(points) * rng.uniform(0.05, 0.1)), max(0, len(valid) - 1))
    if count:
        mask[rng.choice(valid, count, replace=False)] = False
    return points, mask, label


def statistics(features):
    # Fingertip separation uses landmarks 4 and 8 in the normalized shape channels.
    separation = np.linalg.norm(features[:, 12:15] - features[:, 24:27], axis=1)
    return np.r_[
        features.mean(axis=0),
        features.std(axis=0),
        features[-1, 63:65],
        separation.mean(),
        separation.min(),
    ]
