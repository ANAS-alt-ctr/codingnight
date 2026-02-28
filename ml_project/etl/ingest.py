"""
ETL Pipeline: Ingests BTS on-time marketing CSV, cleans it,
creates the is_delayed target, and loads it into the database.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RAW_CSV_PATH, CLEANED_TABLE, DB_URL


def get_engine():
    """Return a SQLAlchemy engine. Falls back to SQLite if Postgres is unavailable."""
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[DB] Connected to PostgreSQL.")
        return engine
    except Exception as e:
        print(f"[DB] PostgreSQL unavailable ({e}). Using SQLite fallback.")
        db_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "flight_db.sqlite")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return create_engine(f"sqlite:///{db_path}")


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load, clean and label the raw BTS CSV."""
    print(f"[ETL] Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    print(f"[ETL] Raw shape: {df.shape}")
    print(f"[ETL] Columns: {df.columns.tolist()}")

    # ---- Standardize column names ----
    df.columns = [c.strip().upper() for c in df.columns]

    # ---- Keep only relevant columns ----
    cols_needed = [
        "FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
        "DEP_DELAY", "ARR_DELAY", "DISTANCE",
        "CARRIER_DELAY", "WEATHER_DELAY", "TAXI_OUT", "TAXI_IN"
    ]
    available = [c for c in cols_needed if c in df.columns]
    df = df[available].copy()

    # ---- Drop rows where ARR_DELAY or DEP_DELAY is null ----
    df.dropna(subset=["ARR_DELAY", "DEP_DELAY"], inplace=True)

    # ---- Cast numeric columns ----
    numeric_cols = ["DEP_DELAY", "ARR_DELAY", "DISTANCE",
                    "CARRIER_DELAY", "WEATHER_DELAY", "TAXI_OUT", "TAXI_IN"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["ARR_DELAY", "DEP_DELAY"], inplace=True)

    # ---- Parse date ----
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")

    # ---- Target labels ----
    df["IS_DELAYED"] = (df["ARR_DELAY"] > 15).astype(int)

    # ---- Fill remaining NaN ----
    for col in ["CARRIER_DELAY", "WEATHER_DELAY", "TAXI_OUT", "TAXI_IN"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    print(f"[ETL] Cleaned shape: {df.shape}")
    print(f"[ETL] Delay rate: {df['IS_DELAYED'].mean():.2%}")
    return df


def ingest(csv_path: str = RAW_CSV_PATH) -> pd.DataFrame:
    engine = get_engine()
    df = load_and_clean(csv_path)

    print(f"[ETL] Writing to table '{CLEANED_TABLE}' ...")
    df.to_sql(CLEANED_TABLE, engine, if_exists="replace", index=False, chunksize=10000)
    print(f"[ETL] Done! {len(df):,} rows stored.")
    return df


if __name__ == "__main__":
    ingest()
