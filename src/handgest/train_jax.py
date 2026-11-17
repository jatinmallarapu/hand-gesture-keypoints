"""Train a channels-last Flax temporal CNN on the identical participant folds."""

import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

from . import LABELS
from .dataset import folds, load, vectorize
from .features import STEPS, WIDTH
from .report import evaluate, macro_f1
from .training import arguments, save_report, write_metrics


class TemporalCNN(nn.Module):
    @nn.compact
    def __call__(self, x, train=False):
        for dilation in (1, 2, 4):
            x = nn.Conv(128, (3,), kernel_dilation=(dilation,), padding="SAME")(x)
            x = nn.gelu(nn.LayerNorm()(x))
            x = nn.Dropout(0.2, deterministic=not train)(x)
        return nn.Dense(len(LABELS))(x.mean(axis=1))


@jax.jit
def train_step(state, x, y, mask, weights, key):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, x, train=True, rngs={"dropout": key})
        targets = optax.smooth_labels(jax.nn.one_hot(y, len(LABELS)), 0.1)
        loss = optax.softmax_cross_entropy(logits, targets)
        return (loss * mask * weights[y]).sum() / (mask * weights[y]).sum()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@jax.jit
def infer(state, x):
    return state.apply_fn({"params": state.params}, x).argmax(axis=-1)


def fit(args, fold, partition):
    train, validation, test = partition
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, init = jax.random.split(key)
    model = TemporalCNN()
    params = model.init(init, jnp.zeros((1, STEPS, WIDTH)))["params"]
    steps = (len(train) + args.batch_size - 1) // args.batch_size
    tx = optax.adamw(optax.cosine_decay_schedule(1e-3, steps * args.epochs), weight_decay=1e-4)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    counts = np.bincount([LABELS.index(row.label) for row in train], minlength=len(LABELS))
    if np.any(counts == 0):
        raise ValueError("Each training fold must contain all eight gestures")
    weights = jnp.asarray(len(train) / (len(LABELS) * counts), dtype=jnp.float32)
    vx, vy = vectorize(validation)
    tx, ty = vectorize(test)
    fold.mkdir(parents=True, exist_ok=True)
    best, stale, rows, steady = -1.0, 0, [], []
    cold_ms = None
    for epoch in range(1, args.epochs + 1):
        x, y = vectorize(train, rng)
        order = rng.permutation(len(x))
        losses, times = [], []
        for lo in range(0, len(x), args.batch_size):
            idx = order[lo : lo + args.batch_size]
            # Padding keeps every training batch the same shape to avoid a second JIT compile.
            mask = np.arange(args.batch_size) < len(idx)
            idx = np.pad(idx, (0, args.batch_size - len(idx)), mode="edge")
            key, dropout = jax.random.split(key)
            start = time.perf_counter()
            state, loss = train_step(state, x[idx], y[idx], mask, weights, dropout)
            losses.append(float(loss))
            elapsed = (time.perf_counter() - start) * 1000
            if cold_ms is None:
                cold_ms = elapsed
            else:
                times.append(elapsed)
                steady.append(elapsed)
        score = macro_f1(vy, np.asarray(infer(state, vx)))
        rows.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "validation_f1": score,
                "mean_step_ms": float(np.mean(times)) if times else None,
            }
        )
        if score > best:
            best, stale = score, 0
            (fold / "best.msgpack").write_bytes(serialization.to_bytes(state.params))
        else:
            stale += 1
            if stale == 10:
                break
    write_metrics(fold / "metrics.csv", rows)
    params = serialization.from_bytes(state.params, (fold / "best.msgpack").read_bytes())
    state = state.replace(params=params)
    return {
        **evaluate(ty, np.asarray(infer(state, tx))),
        "validation_f1": best,
        "parameters": int(sum(x.size for x in jax.tree.leaves(params))),
        "first_step_ms": cold_ms,
        "mean_step_ms": float(np.mean(steady)) if steady else None,
    }


def main():
    parser = arguments(__doc__)
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
