"""Share preprocessing, softmax and abstention between the service and webcam."""

import numpy as np

from . import LABELS
from .features import to_features


class Predictor:
    def __init__(self, logits, threshold=0.6):
        self.logits = logits
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be between zero and one")
        self.threshold = threshold

    def predict(self, points, mask=None):
        points = np.asarray(points, dtype=np.float32)
        if mask is None:
            mask = np.isfinite(points).all(axis=(1, 2))
        features = to_features(points, mask)
        if not np.any(features):
            return {
                "label": "no_gesture",
                "confidence": 0.0,
                "top3": [],
                "abstained": True,
                "reason": "no_usable_landmarks",
            }
        out = np.asarray(self.logits(features[None]))[0]
        if out.shape != (len(LABELS),) or not np.isfinite(out).all():
            raise ValueError("Model must return eight finite logits")
        probabilities = np.exp(out - np.max(out))
        probabilities /= probabilities.sum()
        order = np.argsort(-probabilities, kind="stable")[:3]
        top = [(LABELS[i], float(probabilities[i])) for i in order]
        label, confidence = top[0]
        abstained = confidence < self.threshold
        return {
            "label": "no_gesture" if abstained else label,
            "confidence": confidence,
            "top3": top,
            "abstained": abstained,
            "reason": "low_confidence" if abstained else None,
        }


def load_predictor(path, threshold=0.6):
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return Predictor(lambda batch: session.run(["logits"], {"features": batch})[0], threshold)
