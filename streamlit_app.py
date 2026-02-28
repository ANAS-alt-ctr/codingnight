import streamlit as st
import requests
import datetime
import json

# FastAPI backend URL
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="centered"
)

st.title("🛫 Flight Delay Prediction ML Platform")
st.markdown("Enter flight details below to predict the probability of arrival delay (>15 mins).")

# Check backend health
@st.cache_data(ttl=5)
def check_health():
    try:
        res = requests.get(f"{API_URL}/health", timeout=3)
        return res.status_code == 200, res.json()
    except Exception:
        return False, None

is_up, health_data = check_health()
if is_up:
    if health_data.get("model_loaded"):
        st.success("✅ Backend API is online and the ML model is loaded.")
    else:
        st.warning("⚠️ Backend API is online, but NO model is loaded. Have you run the training step?")
else:
    st.error("❌ Backend API is offline. Please make sure `python -m ml_project.api.main` is running.")

st.divider()

# Known airlines from the dataset
airlines = [
    "9E", "AA", "AS", "B6", "DL", "EV", "F9", "G4", "HA",
    "MQ", "NK", "OH", "OO", "QX", "UA", "VX", "WN", "YV", "YX"
]

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
    if not is_up:
        st.error("Cannot predict: Backend API is offline.")
    else:
        # Prepare payload
        payload = {
            "airline": airline,
            "origin": origin,
            "dest": dest,
            "distance": distance,
            "dep_delay": dep_delay,
            "day_of_week": date.weekday(), # 0=Mon, 6=Sun
            "month": date.month,
            "day_of_month": date.day
        }
        
        with st.spinner("Analyzing flight with XGBoost..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                
                if response.status_code == 200:
                    result = response.json()
                    prob = result["delay_probability"]
                    prediction = result["prediction"]
                    
                    st.divider()
                    st.subheader("Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    if prediction == "Delayed":
                        col1.metric("Predicted Status", "🚨 Delayed")
                    else:
                        col1.metric("Predicted Status", "✅ On-Time")
                        
                    col2.metric("Delay Probability", f"{prob * 100:.1f}%")
                    
                    st.progress(min(prob, 1.0))
                    
                    if prob > 0.5:
                        st.error(f"High risk of delay! ({prob*100:.1f}%)")
                    elif prob > 0.3:
                        st.warning(f"Moderate risk of delay. ({prob*100:.1f}%)")
                    else:
                        st.success(f"Flight is likely to arrive on time. ({prob*100:.1f}% delay chance)")
                    
                    st.caption(f"Model used: {result['model_used']} | Generated at: {result['timestamp']}")
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to backend: {e}")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This frontend connects to the FastAPI backend running the final Machine Learning model.
    The model was trained on the U.S. BTS On-Time Performance dataset.
    
    **Stack:**
    - Streamlit (UI)
    - FastAPI (Backend)
    - XGBoost (Model)
    """)
    
    if is_up and health_data.get("model_loaded"):
        st.divider()
        st.subheader("Model Info")
        try:
            info_res = requests.get(f"{API_URL}/model-info")
            if info_res.status_code == 200:
                info = info_res.json()
                st.json(info["metrics"])
        except Exception:
            pass
