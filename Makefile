# Explicit checkpoint paths keep model artifacts out of the working tree.
DATA ?= data/processed
CHECKPOINT ?= runs/torch-tcn/fold-0/best.onnx
EPOCHS ?= 80
.PHONY: extract baseline train-torch train-jax serve demo test smoke

extract:
	python -m handgest.extract --out "$(DATA)"

baseline:
	python -m handgest.baseline --data "$(DATA)"

train-torch:
	python -m handgest.train_torch --model tcn --data "$(DATA)" --out runs/torch-tcn --epochs $(EPOCHS)
	python -m handgest.train_torch --model gru --data "$(DATA)" --out runs/torch-gru --epochs $(EPOCHS)

train-jax:
	python -m handgest.train_jax --data "$(DATA)" --out runs/jax-tcn --epochs $(EPOCHS)

serve:
	HANDGEST_ONNX="$(CHECKPOINT)" uvicorn api.main:app --host 127.0.0.1 --port 8000

demo:
	python scripts/live_demo.py --checkpoint "$(CHECKPOINT)"

test:
	python -m pytest -q

smoke:
	python scripts/synthetic_fixture.py
	python -m handgest.baseline --data runs/synthetic --out runs/synthetic-baseline.json
