import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and columns
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="California House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 California House Price Predictor")
st.write("Fill in the details below to get an estimated house price.")

# ── User Inputs ───────────────────────────────────────────────
st.subheader("📍 Location")
col1, col2 = st.columns(2)
longitude = col1.number_input("Longitude", value=-119.5, format="%.4f")
latitude  = col2.number_input("Latitude",  value=36.5,  format="%.4f")

st.subheader("🏘️ Neighborhood")
housing_median_age = st.slider("Housing Median Age (years)", 1, 52, 20)
ocean_proximity    = st.selectbox("Ocean Proximity", [
    'INLAND', 'NEAR BAY', 'NEAR OCEAN', '<1H OCEAN', 'ISLAND'
])

st.subheader("👥 Population & Households")
col3, col4 = st.columns(2)
population  = col3.number_input("Population",  value=1500, step=100)
households  = col4.number_input("Households",  value=500,  step=10)

st.subheader("🛏️ Rooms")
col5, col6 = st.columns(2)
total_rooms    = col5.number_input("Total Rooms",    value=2000, step=100)
total_bedrooms = col6.number_input("Total Bedrooms", value=400,  step=10)

st.subheader("💰 Income")
median_income = st.slider("Median Income (tens of thousands $)", 0.5, 15.0, 4.0, step=0.1)

# ── Preprocess & Predict ──────────────────────────────────────
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Log transforms
    for col in ['total_rooms', 'total_bedrooms', 'population', 'households']:
        df[col] = np.log(df[col] + 1)

    # Engineered features
    df['rooms_per_household']       = df['total_rooms']    / df['households']
    df['bedrooms_per_room']         = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household']  = df['population']     / df['households']

    # One-hot encode ocean_proximity
    df = df.join(pd.get_dummies(df['ocean_proximity'], dtype=int))
    df = df.drop('ocean_proximity', axis=1)

    # Align columns to training data
    df = df.reindex(columns=model_columns, fill_value=0)

    return df

if st.button("🔮 Predict Price"):
    input_data = {
        'longitude':          longitude,
        'latitude':           latitude,
        'housing_median_age': housing_median_age,
        'total_rooms':        total_rooms,
        'total_bedrooms':     total_bedrooms,
        'population':         population,
        'households':         households,
        'median_income':      median_income,
        'ocean_proximity':    ocean_proximity
    }

    processed = preprocess_input(input_data)
    prediction = model.predict(processed)[0]

    st.markdown("---")
    st.success(f"### 🏡 Estimated House Price: ${prediction:,.0f}")
    st.caption("This estimate is based on 1990 California census data.")