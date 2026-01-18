#!/usr/bin/env python3
"""
Periodic retraining pipeline.

Checks for drift and retrains model if needed.
Can be scheduled via cron or similar.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import mlflow

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from churn_prediction.data_loader import (
    clean_data,
    create_user_features,
    identify_churned_users,
    load_events,
)
from churn_prediction.model import ChurnModel
from churn_prediction.monitoring import DriftMonitor, PerformanceMonitor, should_retrain

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customer_churn_mini.json"
MODEL_PATH = PROJECT_ROOT / "models" / "production"
REFERENCE_PATH = PROJECT_ROOT / "data" / "processed" / "reference_features.parquet"
MONITORING_PATH = PROJECT_ROOT / "monitoring"


def load_and_prepare_data():
    """Load data and create features."""
    print("Loading data...")
    df_raw = load_events(DATA_PATH)
    df_clean = clean_data(df_raw)
    churned = identify_churned_users(df_clean)
    df_features = create_user_features(df_clean, churned)
    return df_features


def check_and_retrain(force: bool = False):
    """
    Check if retraining is needed and retrain if so.

    Args:
        force: Force retraining regardless of drift/performance
    """
    print(f"[{datetime.now().isoformat()}] Starting retrain check...")

    df_features = load_and_prepare_data()

    # check if we have reference data and existing model
    if not REFERENCE_PATH.exists() or not MODEL_PATH.exists():
        print("No existing model/reference found. Training initial model...")
        train_and_save(df_features)
        return

    # load reference data for drift comparison
    reference_df = pd.read_parquet(REFERENCE_PATH)

    # check drift
    drift_monitor = DriftMonitor(reference_df)
    drift_result = drift_monitor.check_drift(
        df_features, output_path=MONITORING_PATH / "drift"
    )
    print(f"Drift check: {drift_result['drift_share']*100:.1f}% columns drifted")

    # check performance (would need labeled data in production)
    perf_monitor = PerformanceMonitor(MONITORING_PATH / "performance")
    perf_result = perf_monitor.check_degradation()

    # decide if retraining needed
    if force or should_retrain(drift_result, perf_result):
        reason = "forced" if force else "drift/performance degradation detected"
        print(f"Retraining triggered: {reason}")
        train_and_save(df_features)
    else:
        print("No retraining needed.")


def train_and_save(df_features):
    """Train model and save artifacts."""
    import pandas as pd

    print("Training new model...")

    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run(run_name=f"retrain-{datetime.now().strftime('%Y%m%d-%H%M')}"):
        model = ChurnModel(model_type="xgboost")
        metrics = model.train(df_features, log_mlflow=True)

        print(f"Metrics: F1={metrics['f1']:.3f}, AUC={metrics['roc_auc']:.3f}")

        # save model
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

        # save reference data for future drift comparison
        REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_features.to_parquet(REFERENCE_PATH)
        print(f"Reference data saved to {REFERENCE_PATH}")

        # log performance
        perf_monitor = PerformanceMonitor(MONITORING_PATH / "performance")
        perf_monitor.log_performance(metrics)


if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(description="Retrain churn model")
    parser.add_argument("--force", action="store_true", help="Force retraining")
    args = parser.parse_args()

    check_and_retrain(force=args.force)
