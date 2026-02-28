"""
Master pipeline runner: ETL → Feature Engineering → Model Training
Run this script to execute the full ML pipeline end-to-end.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from etl.ingest import ingest
from features.build_features import build_features, get_engine, FEATURES_TABLE
from models.train_mlflow import run_training


def run_pipeline(csv_path: str = None):
    print("=" * 60)
    print("  FLIGHT DELAY PREDICTION PIPELINE")
    print("=" * 60)

    # Step 1: ETL
    print("\n📥 STEP 1: Data Ingestion & Cleaning")
    from config import RAW_CSV_PATH
    path = csv_path or RAW_CSV_PATH
    df_clean = ingest(path)

    # Step 2: Feature Engineering
    print("\n🔧 STEP 2: Feature Engineering")
    engine = get_engine()
    df_feat = build_features(df_clean)
    df_feat.to_sql(FEATURES_TABLE, engine, if_exists="replace", index=False, chunksize=10000)
    print(f"[Pipeline] Features stored: {df_feat.shape}")

    # Step 3: Training
    print("\n🤖 STEP 3: Model Training & MLflow Tracking")
    results, best_name, best_model, feature_cols = run_training()

    print("\n" + "=" * 60)
    print(f"  ✅ PIPELINE COMPLETE. Best model: {best_name}")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Start API:   python -m ml_project.api.main")
    print("  2. Test API:    curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{\"airline\": \"AA\", \"origin\": \"JFK\", \"dest\": \"LAX\", \"distance\": 2475, \"dep_delay\": 10, \"day_of_week\": 1, \"month\": 6}'")
    print("  3. View MLflow: mlflow ui (or open http://localhost:5000)")


if __name__ == "__main__":
    run_pipeline()
