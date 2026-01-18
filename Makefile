.PHONY: install lint format test train serve clean

install:
	uv pip install -e ".[dev]"

lint:
	ruff check src/ scripts/ tests/
	black --check src/ scripts/ tests/

format:
	ruff check --fix src/ scripts/ tests/
	black src/ scripts/ tests/

test:
	pytest tests/ -v --cov=src/churn_prediction

train:
	python scripts/train.py

retrain:
	python scripts/retrain.py

serve:
	uvicorn churn_prediction.api:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
