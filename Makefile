.PHONY: setup test splits synthetic demo

setup:
	pip install -e ".[dev]"

test:
	pytest -q

splits:
	python scripts/make_splits.py

synthetic:
	python scripts/make_synthetic.py --out data/synthetic

demo: synthetic
	python scripts/train.py --model time --data data/synthetic --name demo_time
	python scripts/train.py --model xgb  --data data/synthetic --name demo_xgb
	python scripts/train.py --model cnn  --data data/synthetic --name demo_cnn --epochs 5
