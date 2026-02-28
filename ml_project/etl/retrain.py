"""
Automated Retraining Pipeline.
Pulls data from DB, retrains the best model, and updates the saved model artifact.
Can be triggered by cron job or Airflow DAG.
"""
import os
import sys
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from etl.ingest import ingest
from features.build_features import build_features, get_engine
from config import FEATURES_TABLE, RAW_CSV_PATH


def retrain(csv_path: str = None):
    """Full retraining pipeline: re-ingest → re-featurize → re-train."""
    print("=" * 60)
    print(f"  RETRAINING PIPELINE — {datetime.datetime.utcnow().isoformat()}")
    print("=" * 60)

    path = csv_path or RAW_CSV_PATH

    # Step 1: Re-ingest
    print("\n[Retrain] Step 1: Data Ingestion")
    df_clean = ingest(path)

    # Step 2: Re-featurize
    print("\n[Retrain] Step 2: Feature Engineering")
    engine = get_engine()
    df_feat = build_features(df_clean)
    df_feat.to_sql(FEATURES_TABLE, engine, if_exists="replace", index=False, chunksize=10000)
    print(f"[Retrain] Features stored: {df_feat.shape}")

    # Step 3: Re-train
    print("\n[Retrain] Step 3: Model Training")
    from models.train_mlflow import run_training
    results, best_name, best_model, feature_cols = run_training()

    print("\n" + "=" * 60)
    print(f"  ✅ RETRAINING COMPLETE — Best: {best_name}")
    print(f"  ROC-AUC: {results[best_name][1]['roc_auc']:.4f}")
    print("=" * 60)

    return results, best_name


if __name__ == "__main__":
    retrain()
