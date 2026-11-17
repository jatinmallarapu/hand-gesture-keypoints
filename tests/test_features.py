"""Verify that geometry cleanup preserves directional information and labels."""

import numpy as np

from handgest.dataset import Clip, split
from handgest.features import fill_gaps, mirror, to_features


def test_shape_and_wrist_motion():
    points = np.zeros((8, 21, 3), dtype=np.float32)
    points[:, 9, 1] = 0.1
    points[:, :, 0] += np.linspace(0, 0.2, 8)[:, None]
    x = to_features(points, np.ones(8))
    assert x.shape == (32, 132)
    assert np.allclose(x[:, :63], x[0, :63])
    assert np.isclose(x[-1, 63], 2)


def test_gap_filling_and_empty_hand():
    values = np.zeros((3, 21, 3))
    values[1] = np.nan
    values[2] = 2
    assert np.allclose(fill_gaps(values, [1, 0, 1])[1], 1)
    assert not to_features(values, [0, 0, 0]).any()


def test_mirror_swaps_left_and_right():
    points = np.full((2, 21, 3), 0.2)
    mirrored, label = mirror(points, "swipe_left")
    assert label == "swipe_right"
    assert np.allclose(mirrored[..., 0], 0.8)
    assert mirror(mirrored, label)[1] == "swipe_left"
    assert mirror(points, "pinch")[1] == "pinch"


def test_subject_groups_are_disjoint():
    clips = [
        Clip(np.zeros((8, 21, 3)), np.zeros(8), "no_gesture", subject)
        for subject in ("s01", "s02", "s03", "s01")
    ]
    train, val, test = split(clips, "s03", "s02")
    assert [{x.subject for x in fold} for fold in (train, val, test)] == [{"s01"}, {"s02"}, {"s03"}]
