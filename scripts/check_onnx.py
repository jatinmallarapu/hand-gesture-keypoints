"""Check exported inference logits against the validation-best PyTorch checkpoints."""

from pathlib import Path

import numpy as np
import torch

from handgest.features import STEPS, WIDTH
from handgest.models_torch import MODELS
from handgest.predict import load_predictor


def main():
    features = np.random.default_rng(42).normal(size=(2, STEPS, WIDTH)).astype(np.float32)
    for name in ("tcn", "gru"):
        path = Path(f"runs/torch-{name}/fold-0")
        checkpoint = torch.load(path / "best.pt", weights_only=True)
        model = MODELS[name]().eval()
        model.load_state_dict(checkpoint["state_dict"])
        with torch.inference_mode():
            expected = model(torch.from_numpy(features)).numpy()
        predictor = load_predictor(path / "best.onnx")
        np.testing.assert_allclose(predictor.logits(features), expected, rtol=1e-4, atol=1e-5)
        print(f"{name} ONNX logits match PyTorch for a two-clip batch")


if __name__ == "__main__":
    main()
