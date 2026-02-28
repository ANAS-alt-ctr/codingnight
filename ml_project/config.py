"""
Central configuration for the ML Project.
Uses environment variables with sensible local defaults.
"""
import os

# ---- Database ----
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "flight_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "flight_password")
DB_NAME = os.getenv("DB_NAME", "flight_db")
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ---- MLflow ----
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "flight_delay_prediction"

# ---- Data ----
RAW_CSV_PATH = os.getenv("RAW_CSV_PATH", r"c:\Users\anasr\Downloads\hakaton\T_ONTIME_MARKETING.csv")
CLEANED_TABLE = "flights_cleaned"
FEATURES_TABLE = "flights_features"
TRAINING_TABLE = "flights_training"

# ---- Model ----
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "artifacts")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")

# ---- API ----
API_HOST = "0.0.0.0"
API_PORT = 8000
