# Runtime imports ONNX Runtime rather than loading a Python training checkpoint.
FROM python:3.11-slim AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src/ src/
COPY api/ api/
COPY configs/ configs/
RUN python -m pip install --no-cache-dir '.[video]' 'onnxruntime>=1.20,<2'
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]
