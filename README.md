# 🏠 HomeSense AI — California House Price Predictor

A machine learning web app that predicts California median house prices based on neighborhood features.

## Live Demo
🔗 [homesense-ai.streamlit.app](https://abdelouahab1-housing-app-main-app.streamlit.app)

## Overview
This app uses a **Random Forest Regressor** trained on the 1990 California Housing dataset to predict median house values based on 9 input features.

## Model Performance
| Metric | Score |
|--------|-------|
| R² | 0.80 |
| MAE | $48,200 |
| RMSE | $67,300 |

## Features Used
- Geographic coordinates (longitude, latitude)
- Housing median age
- Ocean proximity
- Total rooms & bedrooms
- Population & households
- Median income

## Tech Stack
- **Frontend** — Streamlit + custom HTML/CSS
- **Model** — scikit-learn Random Forest
- **Data** — 1990 California Census (20,433 samples)
- **Deployment** — Streamlit Cloud

