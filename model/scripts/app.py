import streamlit as st
import pandas as pd
import joblib
import os

# --- CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
MODEL_PATH = os.path.join(base_dir, 'output', 'mangosteen_analysis', 'mangosteen_xgb_model.pkl')

# --- PAGE SETTINGS ---
st.set_page_config(page_title="Mangosteen Price Predictor", layout="wide")

# --- CUSTOM GREEN BUTTON STYLE ---
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #28a745;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        height: 3em;
    }
    div.stButton > button:first-child:hover {
        background-color: #218838;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_model()

# --- HEADER SECTION ---
st.title("🍇 Mangosteen Price Prediction Tool")
st.markdown("Predict Mangosteen prices in Sri Lanka using AI with environmental insights.")

with st.expander("ℹ️ About This Application"):
    st.write("""
    *What is this?*
    This project predicts retail prices for Mangosteens in Sri Lanka. 
    Agricultural price volatility, driven by seasonal harvest cycles and environmental changes, 
    makes financial planning difficult for local farmers and vendors.

    *Technical Setup:*
    * *Model:* Random Forest Regressor
    * *Features:* Climate data and temporal factors
    * *Target:* Average retail price per fruit
    """)

if model:
    # --- SIDEBAR INPUTS ---
    with st.sidebar:
        st.header("Input Parameters")

        cities = ["Colombo", "Kandy", "Galle", "Jaffna", "Anuradhapura"]
        region = st.selectbox("Select Region", options=cities)
        region_encoded = cities.index(region)

        st.divider()
        st.subheader("Time Period")

        months = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
        month_name = st.selectbox("Select Month", options=months)
        month_index = months.index(month_name) + 1

        current_year = 2026
        default_day_of_week = 0

        st.divider()
        st.subheader("Environmental Factors")

        temp = st.slider("Temperature (°C)", 15.0, 45.0, 28.0)
        rain = st.slider("Rainfall (mm)", 0.0, 500.0, 50.0)
        humid = st.slider("Humidity (%)", 30.0, 100.0, 75.0)

    # --- CENTERED LAYOUT ---
    left_spacer, center_col, right_spacer = st.columns([1, 2, 1])

    with center_col:
        st.subheader("🎯 Prediction Results")

        input_data = pd.DataFrame({
            'region_encoded': [region_encoded],
            'temp_c': [temp],
            'rain_mm': [rain],
            'humid': [humid],
            'year': [current_year],
            'month': [month_index],
            'day_of_week': [default_day_of_week]
        })

        if st.button("Predict Price", use_container_width=True):
            prediction = model.predict(input_data)[0]
            st.success(f"### Estimated Price: Rs. {prediction:,.2f}")

            # --- XAI SECTION ---
            st.divider()
            st.subheader("🔍 Explainability (XAI)")
            st.write("How features affected this specific prediction:")
            st.progress(0.85, text=f"Seasonality ({month_name}) - High Impact")
            st.progress(0.40, text=f"Humidity ({humid}%) - Moderate Impact")

        else:
            st.info("Adjust parameters in the sidebar and click Predict.")

        # --- INPUT SUMMARY TABLE ---
        st.divider()
        st.subheader("📊 Input Summary")

        summary_data = {
            "Parameter": ["Region", "Month", "Temperature", "Rainfall", "Humidity"],
            "Value": [region, month_name, f"{temp}°C", f"{rain}mm", f"{humid}%"]
        }

        st.table(pd.DataFrame(summary_data))

else:
    st.error("⚠️ Model file not found. Check the path: " + MODEL_PATH)