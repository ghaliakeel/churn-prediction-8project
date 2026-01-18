#!/usr/bin/env python3
"""
Initial model training script.
"""
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from churn_prediction.data_loader import (
    clean_data,
    create_user_features,
    identify_churned_users,
    load_events,
)
from churn_prediction.model import ChurnModel, analyze_errors

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "customer_churn_mini.json"
MODEL_PATH = PROJECT_ROOT / "models" / "production"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"


def main():
    print("Loading and processing data...")
    df_raw = load_events(DATA_PATH)
    print(f"  Raw events: {len(df_raw):,}")

    df_clean = clean_data(df_raw)
    print(f"  After cleaning: {len(df_clean):,}")

    churned_users = identify_churned_users(df_clean)
    print(f"  Churned users: {len(churned_users)}")

    df_features = create_user_features(df_clean, churned_users)
    print(f"  Total users: {len(df_features)}")
    print(f"  Churn rate: {df_features['churned'].mean()*100:.1f}%")

    # save processed features
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(PROCESSED_PATH / "reference_features.parquet")

    # setup mlflow
    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment("churn-prediction")

    print("\nTraining model...")
    with mlflow.start_run(run_name="initial-training"):
        model = ChurnModel(model_type="xgboost")
        metrics = model.train(df_features, log_mlflow=True)

        print("\nResults:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.3f}")

        print(f"\nConfusion Matrix:")
        cm = metrics["confusion_matrix"]
        print(f"  TN: {cm[0][0]}  FP: {cm[0][1]}")
        print(f"  FN: {cm[1][0]}  TP: {cm[1][1]}")

        # error analysis
        print("\nAnalyzing errors...")
        errors = analyze_errors(model, df_features)
        print(f"  False Positives: {len(errors['false_positives'])}")
        print(f"  False Negatives: {len(errors['false_negatives'])}")

        # save model
        model.save(MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH / "model.joblib"))
        print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
