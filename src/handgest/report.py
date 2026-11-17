"""Use the fixed class list so absent gestures cannot silently inflate macro F1."""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from . import LABELS

CLASSES = list(range(len(LABELS)))


def macro_f1(expected, predicted):
    return float(f1_score(expected, predicted, labels=CLASSES, average="macro", zero_division=0))


def evaluate(expected, predicted):
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": macro_f1(expected, predicted),
        "confusion": confusion_matrix(expected, predicted, labels=CLASSES).tolist(),
        "per_class": classification_report(
            expected,
            predicted,
            labels=CLASSES,
            target_names=LABELS,
            zero_division=0,
            output_dict=True,
        ),
        "examples": len(expected),
    }


def aggregate(rows):
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in ("accuracy", "macro_f1")
    }
