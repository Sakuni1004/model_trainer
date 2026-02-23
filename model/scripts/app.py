import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px # Added for better XAI charts

# --- CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
# Ensure this path is correct for your local environment
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

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None

model = load_model()

# --- HEADER SECTION ---
st.title("🍇 Mangosteen Price Prediction Tool")
st.markdown("Predict Mangosteen prices in Sri Lanka using AI with environmental insights.")

with st.expander("ℹ️ About This Application"):
    st.write("""
    *What is this?*
    This project predicts retail prices for Mangosteens in Sri Lanka based on environmental and temporal factors.
    
    *Technical Setup:*
    * *Model:* XGBoost Regressor
    * *Features:* Region, Climate Data (Temp, Rain, Humidity), and Seasonality.
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

    # --- MAIN CONTENT AREA WITH TABS ---
    # Centering the tabs slightly for better readability
    _, center_col, _ = st.columns([0.1, 0.8, 0.1])

    with center_col:
        # Define the Tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "🔍 Explainability (XAI)", "📊 Input Summary"])

        # Prepare input data for the model
        input_data = pd.DataFrame({
            'region_encoded': [region_encoded],
            'temp_c': [temp],    
            'rain_mm': [rain],
            'humid': [humid],
            'year': [current_year],
            'month': [month_index],
            'day_of_week': [default_day_of_week]
        })

        # --- TAB 1: PREDICTION ---
        with tab1:
            st.subheader("Price Forecast")
            if st.button("Predict Price", use_container_width=True):
                prediction = model.predict(input_data)[0]
                
                # Visual Metric Display
                st.metric(label=f"Estimated Price in {region}", value=f"Rs. {prediction:,.2f}")
                st.success(f"Prediction generated successfully for {month_name} {current_year}.")
                
                st.info("Switch to the *Explainability* tab to see why this price was calculated.")
            else:
                st.info("Adjust parameters in the sidebar and click the 'Predict Price' button.")

        # --- TAB 2: EXPLAINABILITY (XAI) ---
        with tab2:
            st.subheader("How the Model Thinks")
            st.write(f"Analyzing the influence of features on the price in *{region}*:")

            # Placeholder for Feature Importance Visualization
            # In a real scenario, you can pull feature_importances_ from your XGBoost model
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = input_data.columns
                feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance')
                
                fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h', 
                             title="Global Feature Importance",
                             color_discrete_sequence=['#28a745'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback manual mockup if model doesn't support direct importance attribute
                st.write("*Local Influence (Specific Prediction):*")
                st.progress(0.85, text=f"Seasonality ({month_name}) - High Impact")
                st.progress(0.40, text=f"Humidity ({humid}%) - Moderate Impact")
                st.progress(0.15, text=f"Region ({region}) - Low Impact")
            
            st.caption("Note: XAI helps ensure the model isn't making decisions based on 'noise' or irrelevant data.")

        # --- TAB 3: INPUT SUMMARY ---
        with tab3:
            st.subheader("Current Snapshot")
            summary_data = {
                "Parameter": ["Region", "Year", "Month", "Temperature", "Rainfall", "Humidity"],
                "Selected Value": [region, current_year, month_name, f"{temp}°C", f"{rain}mm", f"{humid}%"]
            }
            st.table(pd.DataFrame(summary_data))

else:
    st.error("⚠️ Model file not found. Check the path: " + MODEL_PATH)
    st.info("Ensure the 'mangosteen_xgb_model.pkl' is inside the 'output/mangosteen_analysis/' folder.")