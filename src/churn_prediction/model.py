"""
Model training and evaluation.
"""
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# try to import xgboost, fall back if not available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None


FEATURE_COLS = [
    "n_sessions",
    "n_songs",
    "n_thumbs_up",
    "n_thumbs_down",
    "n_add_playlist",
    "n_add_friend",
    "n_errors",
    "n_help",
    "n_downgrade",
    "n_adverts",
    "days_active",
    "total_listen_time",
    "songs_per_session",
    "thumbs_ratio",
    "is_paid",
    "is_male",
]


class ChurnModel:
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = FEATURE_COLS

    def _get_model(self, weight_ratio: float = 1.0):
        if self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            return XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=weight_ratio,
                random_state=42,
                eval_metric="logloss",
            )
        elif self.model_type == "xgboost":
            # fallback to gradient boosting
            print("XGBoost not available, using GradientBoosting instead")
            self.model_type = "gradient_boosting"
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
            )
        else:
            # random forest as another option
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                class_weight="balanced",
                random_state=42,
            )

    def train(self, df: pd.DataFrame, test_size: float = 0.2, log_mlflow: bool = True):
        X = df[self.feature_cols].copy()
        y = df["churned"].values

        # calculate class weight for imbalance
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        weight_ratio = n_neg / max(n_pos, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # train
        self.model = self._get_model(weight_ratio)
        self.model.fit(X_train_scaled, y_train)

        # evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "accuracy": (y_pred == y_test).mean(),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "class_distribution": {"train_pos": int(y_train.sum()), "train_neg": int((y_train == 0).sum())},
        }

        if log_mlflow:
            self._log_mlflow(metrics, df)

        return metrics

    def _log_mlflow(self, metrics, df):
        mlflow.log_param("model_type", self.model_type)
        mlflow.log_param("n_features", len(self.feature_cols))
        mlflow.log_param("n_samples", len(df))

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)

        # feature importance
        if hasattr(self.model, "feature_importances_"):
            imp = dict(zip(self.feature_cols, self.model.feature_importances_.tolist()))
            mlflow.log_dict(imp, "feature_importance.json")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].copy()
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, path / "model.joblib")
        joblib.dump(self.scaler, path / "scaler.joblib")

        meta = {"model_type": self.model_type, "feature_cols": self.feature_cols}
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)

        with open(path / "metadata.json") as f:
            meta = json.load(f)

        obj = cls(model_type=meta["model_type"])
        obj.model = joblib.load(path / "model.joblib")
        obj.scaler = joblib.load(path / "scaler.joblib")
        obj.feature_cols = meta["feature_cols"]
        return obj


def analyze_errors(model: ChurnModel, df: pd.DataFrame):
    """Check where the model is making mistakes."""
    X = df[model.feature_cols].copy()
    X_scaled = model.scaler.transform(X)

    y_true = df["churned"].values
    y_pred = model.model.predict(X_scaled)
    y_proba = model.model.predict_proba(X_scaled)[:, 1]

    df_out = df.copy()
    df_out["predicted"] = y_pred
    df_out["proba"] = y_proba
    df_out["correct"] = y_true == y_pred

    # false positives and negatives
    fp = df_out[(y_pred == 1) & (y_true == 0)]
    fn = df_out[(y_pred == 0) & (y_true == 1)]

    return {
        "false_positives": fp,
        "false_negatives": fn,
        "fp_summary": fp[model.feature_cols].describe() if len(fp) > 0 else None,
        "fn_summary": fn[model.feature_cols].describe() if len(fn) > 0 else None,
    }
