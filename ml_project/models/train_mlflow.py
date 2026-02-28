"""
Model Training with MLflow Tracking.
Trains Logistic Regression, Random Forest, and XGBoost.
Logs all metrics to MLflow and saves the best model.
"""
import os
import sys
import warnings
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    FEATURES_TABLE, DB_URL, MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME, MODEL_DIR, BEST_MODEL_PATH, PREPROCESSOR_PATH
)

# Numeric features to use for training
FEATURE_COLS = [
    "DEP_DELAY", "DISTANCE", "AIRLINE_CODE",
    "DAY_OF_WEEK", "MONTH", "IS_WEEKEND",
    "DISTANCE_BUCKET", "ORIGIN_FREQ", "DEST_FREQ"
]
TARGET_COL = "IS_DELAYED"


def get_engine():
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception:
        db_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "flight_db.sqlite")
        return create_engine(f"sqlite:///{db_path}")


def load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    engine = get_engine()
    print(f"[Train] Loading features from '{FEATURES_TABLE}' ...")
    df = pd.read_sql(f"SELECT * FROM {FEATURES_TABLE}", engine)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    print(f"[Train] Using features: {available_features}")

    X = df[available_features].copy()
    y = df[TARGET_COL].copy()

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    print(f"[Train] Dataset: {X.shape[0]:,} rows, {X.shape[1]} features, delay_rate={y.mean():.2%}")
    return X, y, available_features


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc_auc = 0.0

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
    }


def train_logistic_regression(X_train, y_train, X_test, y_test):
    print("\n[Train] Training Logistic Regression ...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=200, solver="saga", random_state=42, n_jobs=-1, tol=1e-3
        ))
    ])
    pipe.fit(X_train, y_train)
    metrics = evaluate_model(pipe, X_test, y_test)
    print(f"  -> ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")
    return pipe, metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    print("[Train] Training Random Forest ...")
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=10,
        random_state=42, n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    metrics = evaluate_model(clf, X_test, y_test)
    print(f"  -> ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")
    return clf, metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    print("[Train] Training XGBoost ...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42, n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    metrics = evaluate_model(clf, X_test, y_test)
    print(f"  -> ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")
    return clf, metrics


def run_training():
    X, y, feature_cols = load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[Train] Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Setup MLflow
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        use_mlflow = True
        print(f"[MLflow] Tracking to {MLFLOW_TRACKING_URI}")
    except Exception as e:
        print(f"[MLflow] Unavailable ({e}). Logging locally only.")
        use_mlflow = False
        os.makedirs(os.path.join(os.path.dirname(__file__), "..", "..", "mlruns"), exist_ok=True)
        mlflow.set_tracking_uri(f"file:///{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'mlruns'))}")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    os.makedirs(MODEL_DIR, exist_ok=True)

    results = {}

    # --- Logistic Regression ---
    with mlflow.start_run(run_name="LogisticRegression"):
        mlflow.log_params({"model": "LogisticRegression", "max_iter": 1000,
                           "features": str(feature_cols)})
        lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
        mlflow.log_metrics(lr_metrics)
        mlflow.sklearn.log_model(lr_model, "model")
        results["LogisticRegression"] = (lr_model, lr_metrics)

    # --- Random Forest ---
    with mlflow.start_run(run_name="RandomForest"):
        mlflow.log_params({"model": "RandomForest", "n_estimators": 200,
                           "max_depth": 15, "features": str(feature_cols)})
        rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
        mlflow.log_metrics(rf_metrics)
        mlflow.sklearn.log_model(rf_model, "model")
        results["RandomForest"] = (rf_model, rf_metrics)

    # --- XGBoost ---
    with mlflow.start_run(run_name="XGBoost"):
        mlflow.log_params({"model": "XGBoost", "n_estimators": 300,
                           "max_depth": 7, "lr": 0.05, "features": str(feature_cols)})
        xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
        mlflow.log_metrics(xgb_metrics)
        mlflow.xgboost.log_model(xgb_model, "model")
        results["XGBoost"] = (xgb_model, xgb_metrics)

    # ---- Print comparison ----
    print("\n" + "="*60)
    print("MODEL COMPARISON (Test Set)")
    print("="*60)
    print(f"{'Model':<22} {'Accuracy':>8} {'Precision':>9} {'Recall':>7} {'F1':>6} {'ROC-AUC':>8}")
    print("-"*60)
    for name, (_, m) in results.items():
        print(f"{name:<22} {m['accuracy']:>8.4f} {m['precision']:>9.4f} {m['recall']:>7.4f} {m['f1']:>6.4f} {m['roc_auc']:>8.4f}")
    print("="*60)

    # ---- Select best model by ROC-AUC ----
    best_name = max(results, key=lambda k: results[k][1]["roc_auc"])
    best_model, best_metrics = results[best_name]
    print(f"\n[Train] Best model: {best_name} (ROC-AUC={best_metrics['roc_auc']:.4f})")

    # Save best model + metadata
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": best_model,
            "model_name": best_name,
            "feature_cols": feature_cols,
            "metrics": best_metrics,
        }, f)
    print(f"[Train] Best model saved to: {BEST_MODEL_PATH}")

    # Print classification report for best model
    y_pred_best = best_model.predict(X_test)
    print(f"\n[Train] Classification Report ({best_name}):")
    print(classification_report(y_test, y_pred_best, target_names=["On-Time", "Delayed"]))

    return results, best_name, best_model, feature_cols


if __name__ == "__main__":
    run_training()
