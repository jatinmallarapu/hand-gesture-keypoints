"""Train PyTorch models across subject folds and export the best validation checkpoints."""

import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import LABELS
from .dataset import GestureDataset, folds, load, vectorize
from .features import STEPS, WIDTH
from .models_torch import MODELS
from .report import evaluate, macro_f1
from .training import arguments, save_report, write_metrics


def fit(args, fold, partition):
    train, validation, test = partition
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    model = MODELS[args.model]().to(device)
    loader = DataLoader(
        GestureDataset(train, train=True, seed=args.seed), batch_size=args.batch_size, shuffle=True
    )
    counts = np.bincount([LABELS.index(x.label) for x in train], minlength=len(LABELS))
    if np.any(counts == 0):
        raise ValueError("Each training fold must contain all eight gestures")
    weights = len(train) / (len(LABELS) * counts)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device), label_smoothing=0.1
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    vx, vy = vectorize(validation)
    tx, ty = vectorize(test)

    def predict(x):
        model.eval()
        with torch.inference_mode():
            return model(torch.as_tensor(x, device=device)).argmax(dim=1).cpu().numpy()

    fold.mkdir(parents=True, exist_ok=True)
    best, stale, rows = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, timing = [], []
        for x, y in loader:
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            timing.append((time.perf_counter() - start) * 1000)
        schedule.step()
        score = macro_f1(vy, predict(vx))
        rows.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "validation_f1": score,
                "mean_step_ms": float(np.mean(timing)),
            }
        )
        if score > best:
            best, stale = score, 0
            torch.save(
                {"architecture": args.model, "labels": LABELS, "state_dict": model.state_dict()},
                fold / "best.pt",
            )
        else:
            stale += 1
            if stale == 10:
                break
    write_metrics(fold / "metrics.csv", rows)
    checkpoint = torch.load(fold / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    predictions = predict(tx)
    if not args.skip_onnx:
        # Legacy exporter supports dynamic batch axes and PyTorch's built-in GRU exporter.
        torch.onnx.export(
            model.cpu(),
            torch.zeros((1, STEPS, WIDTH)),
            str(fold / "best.onnx"),
            input_names=["features"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=False,
        )
    return {
        **evaluate(ty, predictions),
        "validation_f1": best,
        "parameters": sum(x.numel() for x in model.parameters()),
        "mean_step_ms": float(np.mean([row["mean_step_ms"] for row in rows])),
    }


def main():
    parser = arguments(__doc__)
    parser.add_argument("--model", choices=MODELS, default="gru")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-onnx", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for subject, validation, samples in folds(load(args.data)):
        result = fit(args, args.out / f"fold-{len(results)}", samples)
        results.append({"test_subject": subject, "validation_subject": validation, **result})
    save_report(args.out / "report.json", results)


if __name__ == "__main__":
    main()
