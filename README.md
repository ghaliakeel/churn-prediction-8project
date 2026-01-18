# Customer Churn Prediction

Predicting which users will cancel their music streaming subscription.

## Problem

Users cancel. We want to know before they do.

## Data

Event logs - every action a user takes: songs played, thumbs up/down, errors, help page visits, ads seen, cancellations.

Defined churn as reaching the `Cancellation Confirmation` page. Misses users who just stop opening the app, but it's unambiguous.

## Features

Aggregated events per user:

- `n_sessions`, `n_songs` - activity level
- `n_thumbs_up`, `n_thumbs_down` - satisfaction
- `n_errors`, `n_help` - frustration signals
- `n_adverts` - free tier annoyance
- `days_active`, `total_listen_time` - engagement
- `songs_per_session`, `thumbs_ratio` - depth
- `is_paid`, `is_male` - user attributes

Only used events before cancellation to avoid leakage.

## Model

GradientBoosting with `scale_pos_weight` for class imbalance (~22% churn rate).

```python
n_estimators=100
max_depth=4
learning_rate=0.1
```

F1: ~82%, AUC: ~96%

## Structure

```
src/churn_prediction/
    data_loader.py    - load and preprocess
    model.py          - training
    api.py            - FastAPI
    monitoring.py     - drift detection

scripts/
    train.py          - training script
    retrain.py        - scheduled retraining

notebooks/
    01_eda.ipynb      - exploration

tests/                - unit tests
```

## Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## Run

```bash
# train
python scripts/train.py

# api
uvicorn churn_prediction.api:app --reload

# tests
pytest tests/
```

## API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"n_sessions": 10, "n_songs": 50, "n_thumbs_up": 5, "n_thumbs_down": 3, "n_add_playlist": 2, "n_add_friend": 1, "n_errors": 4, "n_help": 2, "n_downgrade": 0, "n_adverts": 30, "days_active": 20, "total_listen_time": 15000, "songs_per_session": 5, "thumbs_ratio": 0.6, "is_paid": 0, "is_male": 1}'
```

Response:
```json
{"churn_probability": 0.73, "risk_level": "high"}
```

Endpoints: `/predict` (single), `/predict/batch` (multiple), `/health`

## Docker

```bash
docker-compose up
```

Runs API on port 8000, MLflow on port 5000.

## MLflow

Experiments tracked automatically during training. View:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Monitoring

Evidently for drift detection. `retrain.py` checks drift and retrains if needed.

## Gaps

- No temporal features
- Churn = explicit cancellation only
- No hyperparameter tuning
