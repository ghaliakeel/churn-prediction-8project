# Technical Documentation

## Data

JSON lines format. Each line = one event.

```json
{
  "ts": 1538352117000,
  "userId": "30",
  "page": "NextSong",
  "level": "paid"
}
```

Key pages: `NextSong`, `Thumbs Up`, `Thumbs Down`, `Error`, `Help`, `Roll Advert`, `Cancellation Confirmation`.

## Preprocessing

- Drop rows without userId
- Convert timestamps
- Filter events before cancellation (avoid leakage)

## Churn Definition

User reached `Cancellation Confirmation` = churned.

## Features

**Activity:** n_sessions, n_songs, n_thumbs_up, n_thumbs_down, n_errors, n_help, n_downgrade, n_adverts

**Engagement:** days_active, total_listen_time, songs_per_session, thumbs_ratio

**User:** is_paid, is_male

## Model

GradientBoosting.

```python
n_estimators=100
max_depth=4
learning_rate=0.1
```

Class imbalance (~22% churn): handled with `scale_pos_weight`.

Evaluated on F1 and AUC. Accuracy is misleading with imbalanced data.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |

Returns probability and risk level (low/medium/high).

## Monitoring

Evidently compares incoming data to training data. Flags drift.

`retrain.py` checks drift and performance, retrains if needed.

## Gaps

- No temporal features
- Churn = explicit cancellation only
- No tuning done
