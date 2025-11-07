#!/usr/bin/env python3
"""
Simple Example: Using the Deployed Model

This script demonstrates how to use the exported model bundle
for predictions without running the API server.
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from tensorflow import keras
import joblib
from datetime import datetime, timedelta

# Paths
BUNDLE_DIR = Path(__file__).parent / 'model_bundle'

print("="*80)
print("SIMPLE MODEL USAGE EXAMPLE")
print("="*80)
print()

# ==========================================
# Step 1: Load Components
# ==========================================
print("Step 1: Loading Model Components")
print("-"*80)

# Load model
model = keras.models.load_model(str(BUNDLE_DIR / 'climate_lstm_model.keras'))
print(f"✓ Model loaded")

# Load scalers
feature_scaler = joblib.load(str(BUNDLE_DIR / 'lstm_scaler.pkl'))
cleaning_scalers = joblib.load(str(BUNDLE_DIR / 'cleaning_scalers.pkl'))
print(f"✓ Scalers loaded")

# Load configuration
with open(BUNDLE_DIR / 'model_configuration.json', 'r') as f:
    config = json.load(f)
print(f"✓ Configuration loaded")

print()

# ==========================================
# Step 2: Prepare Sample Data
# ==========================================
print("Step 2: Preparing Sample Data")
print("-"*80)

# Example: Create sample data for one location
# In real use, this would come from NASA POWER API
sample_data = {
    'T2M': 25.5,
    'T2M_MAX': 32.0,
    'T2M_MIN': 18.0,
    'PRECTOTCORR': 45.5,
    'RH2M': 65.0,
    'ALLSKY_SFC_SW_DWN': 185.0,
    'latitude': 28.6139,
    'longitude': 77.2090,
    'month': 11,
    'year': 2024
}

df = pd.DataFrame([sample_data])

# Calculate derived features
df['T2M_range'] = df['T2M_MAX'] - df['T2M_MIN']
df['heat_index'] = df['T2M'] + (0.5 * df['RH2M'] / 100)
df['precip_log'] = np.log1p(df['PRECTOTCORR'])

# Temporal features
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['year_norm'] = 0.5  # Normalized

# Season encoding (November = Fall)
df['season_Fall'] = 1
df['season_Spring'] = 0
df['season_Summer'] = 0
df['season_Winter'] = 0

# Apply cleaning scalers to continuous features
for feature in config['continuous_features']:
    if feature in cleaning_scalers:
        df[f'{feature}_transformed'] = cleaning_scalers[feature].transform(df[[feature]])
    else:
        df[f'{feature}_transformed'] = df[feature]

print(f"✓ Sample data prepared")
print(f"  Location: New Delhi ({sample_data['latitude']}, {sample_data['longitude']})")
print(f"  Month: {sample_data['month']} (November)")
print(f"  Temperature: {sample_data['T2M']}°C")
print(f"  Precipitation: {sample_data['PRECTOTCORR']} mm")
print()

# ==========================================
# Step 3: Make Prediction
# ==========================================
print("Step 3: Making Prediction")
print("-"*80)

# Extract features in correct order
X = df[config['all_features_transformed']].values

# Scale features
X_scaled = feature_scaler.transform(X)

# Reshape for LSTM
X_lstm = X_scaled.reshape((1, 1, 18))

# Predict
prediction = model.predict(X_lstm, verbose=0)

print(f"✓ Prediction complete")
print()
print(f"Results:")
print(f"  Temperature Anomaly:   {prediction[0, 0]:>7.4f}°C")
print(f"  Precipitation Anomaly: {prediction[0, 1]:>7.4f} mm")
print()

# ==========================================
# Interpretation
# ==========================================
print("Interpretation:")
print("-"*80)

temp_anom = prediction[0, 0]
precip_anom = prediction[0, 1]

if abs(temp_anom) < 0.1:
    temp_status = "Normal"
elif temp_anom > 0:
    temp_status = "Warmer than average"
else:
    temp_status = "Cooler than average"

if abs(precip_anom) < 0.1:
    precip_status = "Normal"
elif precip_anom > 0:
    precip_status = "More precipitation than average"
else:
    precip_status = "Less precipitation than average"

print(f"  Temperature: {temp_status}")
print(f"  Precipitation: {precip_status}")
print()

print("="*80)
print("Example complete!")
print("="*80)
print()
print("Next steps:")
print("  1. Fetch real data from NASA POWER API")
print("  2. Process multiple months at once")
print("  3. Deploy as API using deployment/api/api_server.py")
print()
