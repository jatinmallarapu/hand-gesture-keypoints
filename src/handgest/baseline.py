"""Evaluate inexpensive classifiers on identical held-out participant folds."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import folds, load, vectorize
from .features import statistics
from .report import aggregate, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("runs/baseline.json"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    results = {"logistic": [], "random_forest": []}
    for subject, _, (train, _, test) in folds(load(args.data)):
        x, y = vectorize(train)
        tx, ty = vectorize(test)
        x, tx = (np.stack([statistics(row) for row in block]) for block in (x, tx))
        models = {
            "logistic": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed),
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=args.seed
            ),
        }
        for name, model in models.items():
            model.fit(x, y)
            results[name].append({"test_subject": subject, **evaluate(ty, model.predict(tx))})
    output = {key: {"summary": aggregate(rows), "folds": rows} for key, rows in results.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: value["summary"] for key, value in output.items()}, indent=2))


if __name__ == "__main__":
    main()
