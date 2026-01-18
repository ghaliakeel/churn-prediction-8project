"""Tests for churn prediction model."""
import pandas as pd
import pytest

from churn_prediction.model import ChurnModel, FEATURE_COLS


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        "userId": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        "n_sessions": [10, 5, 20, 3, 15, 8, 12, 4, 25, 6],
        "n_songs": [100, 30, 200, 20, 150, 80, 120, 25, 300, 50],
        "n_thumbs_up": [20, 5, 40, 2, 30, 15, 25, 3, 60, 10],
        "n_thumbs_down": [2, 8, 1, 5, 3, 4, 2, 6, 1, 7],
        "n_add_playlist": [10, 2, 15, 1, 12, 8, 10, 1, 20, 3],
        "n_add_friend": [3, 0, 5, 0, 4, 2, 3, 0, 6, 1],
        "n_errors": [1, 5, 0, 4, 1, 2, 1, 3, 0, 4],
        "n_help": [0, 3, 0, 2, 1, 1, 0, 2, 0, 2],
        "n_downgrade": [0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
        "n_adverts": [20, 50, 10, 60, 15, 30, 20, 55, 5, 45],
        "days_active": [30, 10, 45, 7, 35, 25, 30, 8, 50, 15],
        "total_listen_time": [25000, 7500, 50000, 5000, 37500, 20000, 30000, 6250, 75000, 12500],
        "songs_per_session": [10, 6, 10, 6.7, 10, 10, 10, 6.25, 12, 8.3],
        "thumbs_ratio": [0.9, 0.38, 0.98, 0.29, 0.91, 0.79, 0.93, 0.33, 0.98, 0.59],
        "is_paid": [1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
        "is_male": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        "churned": [0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
    }
    return pd.DataFrame(data)


def test_model_train(sample_data):
    """Test model training."""
    model = ChurnModel(model_type="xgboost")
    metrics = model.train(sample_data, test_size=0.3, log_mlflow=False)

    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert model.model is not None


def test_model_predict(sample_data):
    """Test model prediction."""
    model = ChurnModel(model_type="xgboost")
    model.train(sample_data, test_size=0.3, log_mlflow=False)

    predictions = model.predict(sample_data)

    assert len(predictions) == len(sample_data)
    assert all(0 <= p <= 1 for p in predictions)


def test_model_save_load(sample_data, tmp_path):
    """Test model save and load."""
    model = ChurnModel(model_type="xgboost")
    model.train(sample_data, test_size=0.3, log_mlflow=False)

    # save
    save_path = tmp_path / "model"
    model.save(save_path)

    # load
    loaded_model = ChurnModel.load(save_path)

    # compare predictions
    orig_pred = model.predict(sample_data)
    loaded_pred = loaded_model.predict(sample_data)

    assert list(orig_pred) == list(loaded_pred)
