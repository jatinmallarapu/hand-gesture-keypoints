"""Test the HTTP contract without camera hardware or a trained checkpoint."""

import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from handgest.predict import Predictor


def test_unavailable_and_ready(monkeypatch, tmp_path):
    monkeypatch.setenv("HANDGEST_ONNX", str(tmp_path / "missing.onnx"))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "model_ready": False}
        frames = np.random.default_rng(7).uniform(0.2, 0.8, (8, 21, 3)).tolist()
        assert client.post("/predict/keypoints", json={"frames": frames}).status_code == 503
        app.state.predictor = Predictor(lambda _: [[10, 0, 0, 0, 0, 0, 0, 0]])
        data = client.post("/predict/keypoints", json={"frames": frames}).json()
        assert data["label"] == "swipe_left"
        assert len(data["top3"]) == 3
        assert data["latency_ms"] >= 0
        assert client.post("/predict/keypoints", json={"frames": frames[:7]}).status_code == 422


def test_abstain_for_ambiguous_or_missing_frames():
    points = np.random.default_rng(7).uniform(0.2, 0.8, (8, 21, 3))
    predictor = Predictor(lambda _: np.zeros((1, 8)))
    assert predictor.predict(points)["reason"] == "low_confidence"
    result = predictor.predict(np.full((8, 21, 3), np.nan))
    assert result["label"] == "no_gesture"
    assert result["reason"] == "no_usable_landmarks"
