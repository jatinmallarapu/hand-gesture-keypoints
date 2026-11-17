"""Serve bounded video and keypoint requests with explicit model readiness."""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

import yaml
from fastapi import FastAPI, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, FiniteFloat

from handgest.extract import extract_video
from handgest.predict import load_predictor

Point = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
Frame = Annotated[list[Point], Field(min_length=21, max_length=21)]
MAX_UPLOAD = 12 * 1024 * 1024


class KeypointRequest(BaseModel):
    frames: Annotated[list[Frame | None], Field(min_length=8, max_length=300)]


class Prediction(BaseModel):
    label: str
    confidence: float
    top3: list[tuple[str, float]]
    latency_ms: float
    abstained: bool
    reason: str | None


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.environ.get("HANDGEST_ONNX", "runs/torch-tcn/fold-0/best.onnx"))
    config = Path(os.environ.get("HANDGEST_CONFIG", "configs/default.yaml"))
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    app.state.predictor = (
        load_predictor(path, settings["confidence_threshold"]) if path.is_file() else None
    )
    yield


app = FastAPI(title="Hand Gesture Keypoints", lifespan=lifespan)


def get_predictor(request):
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(503, "Set HANDGEST_ONNX to a trained checkpoint")
    return predictor


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_ready": request.app.state.predictor is not None}


@app.post("/predict/keypoints", response_model=Prediction)
def predict_keypoints(data: KeypointRequest, request: Request):
    predictor = get_predictor(request)
    start = time.perf_counter()
    points = [
        frame if frame is not None else [[float("nan")] * 3 for _ in range(21)]
        for frame in data.frames
    ]
    result = predictor.predict(points)
    return {**result, "latency_ms": (time.perf_counter() - start) * 1000}


@app.post("/predict/video", response_model=Prediction)
def predict_video(file: UploadFile, request: Request):
    predictor = get_predictor(request)
    start = time.perf_counter()
    model = os.environ.get("HANDGEST_LANDMARKER", "models/hand_landmarker.task")
    if not Path(model).is_file():
        raise HTTPException(503, "MediaPipe model asset is missing")
    with TemporaryDirectory() as storage:
        path = Path(storage, "clip.mp4")
        count = 0
        with path.open("wb") as destination:
            while chunk := file.file.read(1024 * 1024):
                count += len(chunk)
                if count > MAX_UPLOAD:
                    raise HTTPException(413, "Video exceeds 12 MiB")
                destination.write(chunk)
        try:
            points, mask, _ = extract_video(path, model)
            if len(points) < 8:
                raise ValueError("Need at least eight frames")
        except ValueError:
            raise HTTPException(422, "Unsupported, empty or over-limit video") from None
        result = predictor.predict(points, mask)
    return {**result, "latency_ms": (time.perf_counter() - start) * 1000}
