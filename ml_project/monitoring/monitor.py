"""
Model Monitoring: Data Drift Detection using Evidently AI.
Compares training data distribution against new inference data.
Logs inference requests and prediction distributions.
"""
import os
import sys
import json
import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
INFERENCE_LOG = os.path.join(LOG_DIR, "inference_log.jsonl")
DRIFT_REPORT_DIR = os.path.join(LOG_DIR, "drift_reports")


def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(DRIFT_REPORT_DIR, exist_ok=True)


def log_inference(input_data: dict, prediction: dict):
    """Append an inference request + result to the log file."""
    ensure_dirs()
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "input": input_data,
        "prediction": prediction,
    }
    with open(INFERENCE_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def load_inference_log() -> pd.DataFrame:
    """Load inference log into a DataFrame."""
    if not os.path.exists(INFERENCE_LOG):
        return pd.DataFrame()
    records = []
    with open(INFERENCE_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return pd.DataFrame()

    rows = []
    for r in records:
        row = {**r.get("input", {}), **r.get("prediction", {}), "timestamp": r["timestamp"]}
        rows.append(row)
    return pd.DataFrame(rows)


def get_prediction_stats() -> dict:
    """Get summary statistics of prediction distribution."""
    df = load_inference_log()
    if df.empty:
        return {"total_predictions": 0}

    stats = {
        "total_predictions": len(df),
        "delayed_rate": float((df.get("prediction", pd.Series()) == "Delayed").mean()) if "prediction" in df.columns else 0,
        "avg_delay_probability": float(df["delay_probability"].mean()) if "delay_probability" in df.columns else 0,
        "std_delay_probability": float(df["delay_probability"].std()) if "delay_probability" in df.columns else 0,
    }
    return stats


def check_data_drift(training_data: pd.DataFrame, inference_data: pd.DataFrame,
                     feature_cols: list) -> dict:
    """
    Simple data drift check using statistical tests.
    Compares feature distributions between training and inference data.
    """
    drift_results = {}
    for col in feature_cols:
        if col not in training_data.columns or col not in inference_data.columns:
            continue

        train_mean = training_data[col].mean()
        train_std = training_data[col].std()
        inf_mean = inference_data[col].mean()
        inf_std = inference_data[col].std()

        # Simple drift metric: normalized mean shift
        if train_std > 0:
            drift_score = abs(inf_mean - train_mean) / train_std
        else:
            drift_score = 0.0

        drift_results[col] = {
            "train_mean": round(float(train_mean), 4),
            "inference_mean": round(float(inf_mean), 4),
            "drift_score": round(float(drift_score), 4),
            "is_drifted": drift_score > 2.0,  # flag if > 2 std deviations
        }

    return drift_results


def run_drift_report():
    """Generate and save a drift report."""
    ensure_dirs()
    from config import FEATURES_TABLE

    # Load training data from DB
    from features.build_features import get_engine
    engine = get_engine()
    train_df = pd.read_sql(f"SELECT * FROM {FEATURES_TABLE}", engine)

    # Load inference log
    inf_df = load_inference_log()

    if inf_df.empty:
        print("[Monitor] No inference data logged yet. Skipping drift check.")
        return

    feature_cols = ["DEP_DELAY", "DISTANCE", "AIRLINE_CODE", "DAY_OF_WEEK", "MONTH"]
    available = [c for c in feature_cols if c in train_df.columns and c in inf_df.columns]

    drift = check_data_drift(train_df, inf_df, available)

    report = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "prediction_stats": get_prediction_stats(),
        "drift_results": drift,
    }

    report_path = os.path.join(DRIFT_REPORT_DIR, f"drift_{datetime.date.today().isoformat()}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[Monitor] Drift report saved to: {report_path}")
    print(f"[Monitor] Prediction stats: {report['prediction_stats']}")

    drifted = [k for k, v in drift.items() if v.get("is_drifted")]
    if drifted:
        print(f"[Monitor] WARNING: Drift detected in features: {drifted}")
    else:
        print("[Monitor] No significant drift detected.")

    return report


if __name__ == "__main__":
    run_drift_report()
