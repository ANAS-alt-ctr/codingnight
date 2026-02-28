import streamlit as st
import datetime
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(__file__))
from ml_project.config import BEST_MODEL_PATH

st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="centered"
)

st.title("🛫 Flight Delay Prediction ML Platform")
st.markdown("Enter flight details below to predict the probability of arrival delay (>15 mins).")

# ---- Model Loading ----
@st.cache_resource
def load_model():
    if os.path.exists(BEST_MODEL_PATH):
        with open(BEST_MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

model_payload = load_model()

if model_payload:
    st.success(f"✅ ML Model loaded successfully: **{model_payload['model_name']}**")
else:
    st.error(f"❌ Could not find model at {BEST_MODEL_PATH}. Please run the training pipeline first.")

st.divider()

# Known airlines from the dataset
airlines = [
    "9E", "AA", "AS", "B6", "DL", "EV", "F9", "G4", "HA",
    "MQ", "NK", "OH", "OO", "QX", "UA", "VX", "WN", "YV", "YX"
]
airline_map = {a: i for i, a in enumerate(sorted(airlines))}

# Create form
with st.form("prediction_form"):
    st.subheader("Flight Information")
    
    col1, col2 = st.columns(2)
    with col1:
        airline = st.selectbox("Airline (Carrier Code)", airlines, index=1)
        origin = st.text_input("Origin Airport (e.g., JFK, ATL)", "JFK").upper()
        dest = st.text_input("Destination Airport (e.g., LAX, ORD)", "LAX").upper()
    
    with col2:
        date = st.date_input("Flight Date", datetime.date.today())
        distance = st.number_input("Distance (miles)", min_value=10, max_value=10000, value=2500)
        dep_delay = st.number_input("Current Departure Delay (mins)", min_value=-60, max_value=800, value=0)
    
    submit_button = st.form_submit_button("Predict Delay")

if submit_button:
    if not model_payload:
        st.error("Cannot predict: Model is not loaded.")
    else:
        with st.spinner("Analyzing flight with ML model..."):
            try:
                # 1. Feature Engineering (Encode the input)
                airline_code = airline_map.get(airline, -1)
                day_of_week = date.weekday()
                month = date.month
                is_weekend = 1 if day_of_week in [5, 6] else 0

                if distance <= 500:
                    dist_bucket = 0
                elif distance <= 1000:
                    dist_bucket = 1
                elif distance <= 2000:
                    dist_bucket = 2
                else:
                    dist_bucket = 3

                # Match feature order from training
                raw_features = {
                    "DEP_DELAY": dep_delay,
                    "DISTANCE": distance,
                    "AIRLINE_CODE": airline_code,
                    "DAY_OF_WEEK": day_of_week,
                    "MONTH": month,
                    "IS_WEEKEND": is_weekend,
                    "DISTANCE_BUCKET": dist_bucket,
                    "ORIGIN_FREQ": 0.05,   # approximation for unseen in demo
                    "DEST_FREQ": 0.05,
                }
                
                feature_cols = model_payload["feature_cols"]
                X = [[raw_features.get(col, 0) for col in feature_cols]]
                
                # 2. Predict
                model = model_payload["model"]
                prob = float(model.predict_proba(X)[0][1])
                prediction = "Delayed" if prob >= 0.5 else "On-Time"
                
                st.divider()
                st.subheader("Prediction Results")
                
                res_col1, res_col2 = st.columns(2)
                
                if prediction == "Delayed":
                    res_col1.metric("Predicted Status", "🚨 Delayed")
                else:
                    res_col1.metric("Predicted Status", "✅ On-Time")
                    
                res_col2.metric("Delay Probability", f"{prob * 100:.1f}%")
                
                st.progress(min(prob, 1.0))
                
                if prob > 0.5:
                    st.error(f"High risk of delay! ({prob*100:.1f}%)")
                elif prob > 0.3:
                    st.warning(f"Moderate risk of delay. ({prob*100:.1f}%)")
                else:
                    st.success(f"Flight is likely to arrive on time. ({prob*100:.1f}% delay chance)")
                
                st.caption(f"Model used: {model_payload['model_name']} | Generated at: {datetime.datetime.now().isoformat()}")
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This frontend is a **standalone** Streamlit application. It loads the saved XGBoost ML model directly and makes predictions without needing a separate FastAPI backend server.
    
    The model was trained on the U.S. BTS On-Time Performance dataset.
    """)
    
    if model_payload:
        st.divider()
        st.subheader("Model Info")
        st.json(model_payload["metrics"])
