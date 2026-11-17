"""Keep experiment output and argument handling identical across frameworks."""

import argparse
import csv
import json
from pathlib import Path

from .report import aggregate


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80, choices=range(1, 81))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def write_metrics(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_report(path, rows):
    report = {"summary": aggregate(rows), "folds": rows}
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
