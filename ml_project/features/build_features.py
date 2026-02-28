"""
Feature Engineering: Reads cleaned flights from DB,
builds ML features, and writes a features table back to DB.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CLEANED_TABLE, FEATURES_TABLE, DB_URL


def get_engine():
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception:
        db_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "flight_db.sqlite")
        return create_engine(f"sqlite:///{db_path}")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML-ready features from cleaned flight data."""
    feat = df.copy()

    # ---- Date-based features ----
    if "FL_DATE" in feat.columns:
        feat["FL_DATE"] = pd.to_datetime(feat["FL_DATE"], errors="coerce")
        feat["DAY_OF_WEEK"] = feat["FL_DATE"].dt.dayofweek        # 0=Mon, 6=Sun
        feat["MONTH"] = feat["FL_DATE"].dt.month
        feat["DAY_OF_MONTH"] = feat["FL_DATE"].dt.day
        feat["IS_WEEKEND"] = feat["DAY_OF_WEEK"].isin([5, 6]).astype(int)

    # ---- Distance buckets ----
    if "DISTANCE" in feat.columns:
        feat["DISTANCE_BUCKET"] = pd.cut(
            feat["DISTANCE"],
            bins=[0, 500, 1000, 2000, 5000],
            labels=[0, 1, 2, 3]  # short, medium, long, ultra
        ).astype(float)

    # ---- Airline label encoding ----
    if "OP_UNIQUE_CARRIER" in feat.columns:
        airlines = sorted(feat["OP_UNIQUE_CARRIER"].dropna().unique())
        airline_map = {a: i for i, a in enumerate(airlines)}
        feat["AIRLINE_CODE"] = feat["OP_UNIQUE_CARRIER"].map(airline_map).fillna(-1).astype(int)

    # ---- Origin / Dest frequency encoding ----
    for col in ["ORIGIN", "DEST"]:
        if col in feat.columns:
            freq = feat[col].value_counts(normalize=True)
            feat[f"{col}_FREQ"] = feat[col].map(freq).fillna(0)

    # ---- Departure delay bins ----
    if "DEP_DELAY" in feat.columns:
        feat["DEP_DELAY"] = feat["DEP_DELAY"].clip(lower=-60, upper=300)

    return feat


def run_feature_engineering():
    engine = get_engine()
    print(f"[Features] Reading from '{CLEANED_TABLE}' ...")
    df = pd.read_sql(f"SELECT * FROM {CLEANED_TABLE}", engine)

    df_feat = build_features(df)
    print(f"[Features] Writing to '{FEATURES_TABLE}' ...")
    df_feat.to_sql(FEATURES_TABLE, engine, if_exists="replace", index=False, chunksize=10000)
    print(f"[Features] Done! {len(df_feat):,} rows with {len(df_feat.columns)} features.")
    return df_feat


if __name__ == "__main__":
    run_feature_engineering()
