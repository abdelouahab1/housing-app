import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="HomeSense AI", page_icon="🏠", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f0f4f8; }
[data-testid="stHeader"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Navbar ── */
.navbar {
    background: #042C53; padding: 0 2rem;
    display: flex; align-items: center; justify-content: space-between;
    height: 56px;
}
.nav-left { display: flex; align-items: center; gap: 10px; }
.nav-icon { width: 32px; height: 32px; background: #1D9E75; border-radius: 8px;
    display: flex; align-items: center; justify-content: center; font-size: 16px; }
.nav-logo { font-size: 16px; font-weight: 500; color: #E6F1FB; }
.nav-right { display: flex; align-items: center; gap: 8px; }
.nav-dot { width: 8px; height: 8px; background: #1D9E75; border-radius: 50%; }
.nav-tag { font-size: 12px; background: #0C447C; color: #85B7EB; padding: 5px 12px; border-radius: 20px; }

/* ── Hero ── */
.hero { background: #185FA5; padding: 2.5rem 2rem 2rem; text-align: center; }
.hero-eyebrow { font-size: 11px; font-weight: 500; letter-spacing: 0.12em;
    color: #85B7EB; text-transform: uppercase; margin-bottom: 10px; }
.hero-title { font-size: 32px; font-weight: 500; color: #E6F1FB; margin-bottom: 8px; }
.hero-sub { font-size: 14px; color: #85B7EB; margin-bottom: 1.25rem; }
.hero-chips { display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; }
.chip { font-size: 12px; padding: 5px 13px; border-radius: 20px; background: #0C447C; color: #B5D4F4; }
.chip-green { background: #085041; color: #9FE1CB; }

/* ── Metrics bar ── */
.metrics-bar { background: #042C53; display: grid; grid-template-columns: repeat(4, 1fr); }
.metric { padding: 18px; text-align: center; border-right: 0.5px solid #0C447C; }
.metric:last-child { border-right: none; }
.metric-val { font-size: 22px; font-weight: 500; color: #9FE1CB; }
.metric-label { font-size: 11px; color: #378ADD; margin-top: 3px; }

/* ── Content wrapper ── */
.content-wrap { max-width: 900px; margin: 0 auto; padding: 1.5rem 2rem; }

/* ── Section headers ── */
.section-header {
    display: flex; align-items: center; gap: 12px;
    background: #E6F1FB; border: 0.5px solid #B5D4F4;
    border-radius: 10px 10px 0 0; padding: 12px 16px;
    margin-top: 20px;
}
.section-icon { width: 28px; height: 28px; background: #185FA5; border-radius: 6px;
    display: flex; align-items: center; justify-content: center; font-size: 14px; }
.section-label { font-size: 14px; font-weight: 500; color: #0C447C; }

/* ── Streamlit container as card body ── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border: 0.5px solid #dbe8f5 !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 1rem 1.25rem !important;
}

/* ── Input styling ── */
[data-testid="stNumberInput"] input {
    background: #f7fafd !important;
    border: 0.5px solid #c8d8ea !important;
    border-radius: 8px !important;
    font-size: 15px !important;
    padding: 10px 12px !important;
    height: 44px !important;
}
[data-baseweb="select"] > div {
    background: #f7fafd !important;
    border: 0.5px solid #c8d8ea !important;
    border-radius: 8px !important;
    min-height: 44px !important;
}
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #444441 !important;
    margin-bottom: 4px !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #185FA5 !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100% !important;
    background: #185FA5 !important;
    color: #E6F1FB !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px !important;
    margin-top: 8px !important;
    height: auto !important;
}
.stButton > button:hover { background: #0C447C !important; }

/* ── Result panel ── */
.result-wrap { border-radius: 12px; overflow: hidden; border: 2px solid #1D9E75; margin-top: 1rem; }
.result-top { background: #085041; padding: 1.5rem;
    display: flex; align-items: center; justify-content: space-between; }
.result-label { font-size: 12px; color: #5DCAA5; margin-bottom: 6px; }
.result-price { font-size: 40px; font-weight: 500; color: #E1F5EE; }
.confidence-label { font-size: 11px; color: #5DCAA5; margin-bottom: 6px; text-align: right; }
.confidence-bar-bg { width: 130px; height: 6px; background: #0F6E56;
    border-radius: 3px; overflow: hidden; margin-left: auto; }
.confidence-bar { height: 100%; width: 80%; background: #1D9E75; border-radius: 3px; }
.confidence-pct { font-size: 14px; color: #9FE1CB; margin-top: 5px; font-weight: 500; text-align: right; }
.result-stats { background: #0F6E56; display: grid; grid-template-columns: repeat(3, 1fr); }
.rs { padding: 14px; text-align: center; border-right: 0.5px solid #085041; }
.rs:last-child { border-right: none; }
.rs-val { font-size: 16px; font-weight: 500; color: #9FE1CB; }
.rs-label { font-size: 11px; color: #5DCAA5; margin-top: 3px; }
.result-footer { background: #fff; padding: 10px 16px; border-top: 0.5px solid #dbe8f5; }
.result-note { font-size: 12px; color: #888780; text-align: center; }

/* ── Footer ── */
.footer { background: #042C53; padding: 1rem 2rem; margin-top: 2rem;
    display: flex; align-items: center; justify-content: space-between; }
.footer-left { font-size: 12px; color: #378ADD; }
.footer-right { font-size: 12px; color: #185FA5; }
</style>
""", unsafe_allow_html=True)

# ── Navbar ───────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="nav-left">
        <div class="nav-icon">🏠</div>
        <span class="nav-logo">HomeSense AI</span>
    </div>
    <div class="nav-right">
        <div class="nav-dot"></div>
        <span class="nav-tag">Model live</span>
    </div>
</div>
<div class="hero">
    <div class="hero-eyebrow">Powered by Random Forest</div>
    <div class="hero-title">California House Price Predictor</div>
    <div class="hero-sub">Enter neighborhood details below to get an instant ML-powered estimate</div>
    <div class="hero-chips">
        <span class="chip">20,433 training samples</span>
        <span class="chip chip-green">R² score: 0.80</span>
        <span class="chip">9 features</span>
        <span class="chip">MAE: $48,200</span>
    </div>
</div>
<div class="metrics-bar">
    <div class="metric"><div class="metric-val">0.80</div><div class="metric-label">R² accuracy</div></div>
    <div class="metric"><div class="metric-val">$48k</div><div class="metric-label">Avg error</div></div>
    <div class="metric"><div class="metric-val">50</div><div class="metric-label">Decision trees</div></div>
    <div class="metric"><div class="metric-val">1990</div><div class="metric-label">Census year</div></div>
</div>
<div class="content-wrap">
""", unsafe_allow_html=True)

# ── Location ─────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-icon">📍</div>
    <span class="section-label">Location details</span>
</div>
""", unsafe_allow_html=True)
with st.container(border=True):
    col1, col2 = st.columns(2)
    longitude = col1.number_input("Longitude", value=-119.57, format="%.4f")
    latitude  = col2.number_input("Latitude",  value=36.49,  format="%.4f")

# ── Neighborhood ─────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-icon">🏘️</div>
    <span class="section-label">Neighborhood info</span>
</div>
""", unsafe_allow_html=True)
with st.container(border=True):
    col3, col4 = st.columns(2)
    ocean_proximity    = col3.selectbox("Ocean proximity", ['INLAND','NEAR BAY','NEAR OCEAN','<1H OCEAN','ISLAND'])
    housing_median_age = col4.slider("Housing median age (years)", 1, 52, 20)

# ── Population & Rooms ───────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-icon">👥</div>
    <span class="section-label">Population & housing</span>
</div>
""", unsafe_allow_html=True)
with st.container(border=True):
    col5, col6, col7 = st.columns(3)
    population  = col5.number_input("Population",    value=1500, step=100)
    households  = col6.number_input("Households",    value=500,  step=10)
    total_rooms = col7.number_input("Total rooms",   value=2000, step=100)
    col8, col9  = st.columns(2)
    total_bedrooms = col8.number_input("Total bedrooms", value=400, step=10)
    median_income  = col9.slider("Median income (×$10k)", 0.5, 15.0, 4.0, step=0.1)

# ── Predict ──────────────────────────────────────────────────────────
def preprocess_input(data):
    df = pd.DataFrame([data])
    for col in ['total_rooms', 'total_bedrooms', 'population', 'households']:
        df[col] = np.log(df[col] + 1)
    df['rooms_per_household']      = df['total_rooms']    / df['households']
    df['bedrooms_per_room']        = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population']     / df['households']
    df = df.join(pd.get_dummies(df['ocean_proximity'], dtype=int))
    df = df.drop('ocean_proximity', axis=1)
    df = df.reindex(columns=model_columns, fill_value=0)
    return df

if st.button("→  Estimate house price"):
    input_data = {
        'longitude': longitude, 'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms, 'total_bedrooms': total_bedrooms,
        'population': population, 'households': households,
        'median_income': median_income, 'ocean_proximity': ocean_proximity
    }
    prediction = model.predict(preprocess_input(input_data))[0]
    st.markdown(f"""
    <div class="result-wrap">
        <div class="result-top">
            <div>
                <div class="result-label">Estimated median house value</div>
                <div class="result-price">${prediction:,.0f}</div>
            </div>
            <div>
                <div class="confidence-label">Model confidence</div>
                <div class="confidence-bar-bg"><div class="confidence-bar"></div></div>
                <div class="confidence-pct">80%</div>
            </div>
        </div>
        <div class="result-stats">
            <div class="rs"><div class="rs-val">±$48,200</div><div class="rs-label">Mean abs. error</div></div>
            <div class="rs"><div class="rs-val">$67,300</div><div class="rs-label">Root mean sq. error</div></div>
            <div class="rs"><div class="rs-val">R² 0.80</div><div class="rs-label">Explained variance</div></div>
        </div>
        <div class="result-footer">
            <div class="result-note">Based on 1990 California census · Random Forest · 50 estimators · Not a real estate appraisal</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("""
</div>
<div class="footer">
    <span class="footer-left">HomeSense AI · Built with Streamlit + scikit-learn</span>
    <span class="footer-right">California Housing Dataset · 1990</span>
</div>
""", unsafe_allow_html=True)