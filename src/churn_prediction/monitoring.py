"""
Drift detection and performance monitoring.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from .model import FEATURE_COLS


class DriftMonitor:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference = reference_data[FEATURE_COLS].copy()
        self.column_mapping = ColumnMapping(
            numerical_features=FEATURE_COLS,
            target=None,
        )

    def check_drift(self, current_data: pd.DataFrame, output_path: Optional[Path] = None) -> dict:
        current = current_data[FEATURE_COLS].copy()

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference,
            current_data=current,
            column_mapping=self.column_mapping,
        )

        result = report.as_dict()

        drift_summary = {
            "timestamp": datetime.now().isoformat(),
            "n_reference_samples": len(self.reference),
            "n_current_samples": len(current),
            "dataset_drift_detected": result["metrics"][0]["result"]["dataset_drift"],
            "drift_share": result["metrics"][0]["result"]["share_of_drifted_columns"],
            "drifted_columns": [],
        }

        for col, col_result in result["metrics"][0]["result"]["drift_by_columns"].items():
            if col_result["drift_detected"]:
                drift_summary["drifted_columns"].append({
                    "column": col,
                    "drift_score": col_result["drift_score"],
                })

        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
            report.save_html(str(output_path / "drift_report.html"))
            with open(output_path / "drift_summary.json", "w") as f:
                json.dump(drift_summary, f, indent=2)

        return drift_summary


class PerformanceMonitor:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.history_file = self.log_path / "performance_history.json"

        if self.history_file.exists():
            with open(self.history_file) as f:
                self.history = json.load(f)
        else:
            self.history = []

    def log_performance(self, metrics: dict):
        entry = {"timestamp": datetime.now().isoformat(), **metrics}
        self.history.append(entry)

        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def check_degradation(self, threshold: float = 0.1) -> dict:
        if len(self.history) < 2:
            return {"degraded": False, "message": "Not enough history"}

        recent = self.history[-1]
        previous = self.history[:-1]

        avg_f1 = sum(h.get("f1", 0) for h in previous) / len(previous)
        recent_f1 = recent.get("f1", 0)

        degraded = (avg_f1 - recent_f1) > threshold

        return {
            "degraded": degraded,
            "recent_f1": recent_f1,
            "historical_avg_f1": avg_f1,
            "drop": avg_f1 - recent_f1,
        }


def should_retrain(drift_result: dict, perf_result: dict) -> bool:
    # retrain if >30% columns drifted or performance dropped
    drift_triggered = drift_result.get("drift_share", 0) > 0.3
    perf_triggered = perf_result.get("degraded", False)
    return drift_triggered or perf_triggered
