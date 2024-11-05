# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 10:18:33 2024

@author: Awarri User
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Boston housing dataset
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns=["MEDV"])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a simple linear regression model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit App UI
st.title("Boston Housing Price Prediction")

# Sidebar for input
st.sidebar.header("Adjust the sliders to predict house price")

# Create sliders for all the numerical features in the Boston dataset
slider_inputs = {}
for feature in X.columns:
    slider_inputs[feature] = st.sidebar.slider(f"{feature}", 
                                               min_value=float(X[feature].min()), 
                                               max_value=float(X[feature].max()), 
                                               value=float(X[feature].mean()), 
                                               step=0.1)

# Convert the slider inputs to a dataframe
input_data = pd.DataFrame([slider_inputs], columns=X.columns)

# Standardize the input data
input_data_scaled = scaler.transform(input_data)

# Predict the price of the house using the trained model
predicted_price = model.predict(input_data_scaled)

# Display the prediction
st.subheader("Predicted House Price (MEDV):")
st.write(f"${predicted_price[0][0]:,.2f}")

# Optional: Show model coefficients (useful for insight)
st.subheader("Model Coefficients:")
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})
st.write(coeff_df)

# Show the actual vs predicted (on test set)
st.subheader("Model Performance (on Test Set)")
st.write(f"R² Score: {model.score(X_test, y_test):.4f}")
