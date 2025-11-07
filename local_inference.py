#!/usr/bin/env python3
"""
Local LSTM Model Inference Script
Adapted from Google Colab workflow for local use

This script loads the trained LSTM model and makes predictions on the transformed data.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tensorflow import keras
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Create results directory
RESULTS_DIR.mkdir(exist_ok=True)

print("="*80)
print("NASA CLIMATE LSTM MODEL - LOCAL INFERENCE")
print("="*80)
print()

# ========================================
# STEP 1: Load Model and Metadata
# ========================================
print("📦 Loading model and metadata...")
print("-"*80)

# Load LSTM model
model_path = MODEL_DIR / 'climate_lstm_model.keras'
model = keras.models.load_model(str(model_path))
print(f"✓ LSTM model loaded from: {model_path.name}")

# Load scaler
scaler_path = MODEL_DIR / 'lstm_scaler.pkl'
scaler = joblib.load(str(scaler_path))
print(f"✓ Scaler loaded from: {scaler_path.name}")

# Load metadata
metadata_path = MODEL_DIR / 'lstm_model_metadata.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
print(f"✓ Metadata loaded from: {metadata_path.name}")

# Load model configuration
config_path = DATA_DIR / 'model_configuration.json'
with open(config_path, 'r') as f:
    model_config = json.load(f)
print(f"✓ Model configuration loaded from: {config_path.name}")

print()
print("📊 Model Information:")
print(f"   Type: {metadata['model_type']}")
print(f"   Input Shape: {model.input_shape}")
print(f"   Output Shape: {model.output_shape}")
print(f"   Features: {metadata['n_features']}")
print(f"   Targets: {metadata['n_targets']}")
print(f"   Test RMSE: {metadata['test_rmse']:.6f}")
print(f"   R² Temperature: {metadata['r2_temperature']:.4f}")
print(f"   R² Precipitation: {metadata['r2_precipitation']:.4f}")
print()

# ========================================
# STEP 2: Load Data
# ========================================
print("📂 Loading data...")
print("-"*80)

data_path = DATA_DIR / 'climate_model_ready_transformed.csv'
df_transformed = pd.read_csv(data_path)
print(f"✓ Data loaded: {df_transformed.shape[0]:,} rows × {df_transformed.shape[1]} columns")

# Prepare features and targets
X = df_transformed[model_config['all_features_transformed']].values
y = df_transformed[model_config['target_variables']].values

print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print()

# ========================================
# STEP 3: Make Predictions
# ========================================
print("🔮 Making predictions...")
print("-"*80)

# Scale features
X_scaled = scaler.transform(X)
print(f"✓ Features scaled")

# Reshape for LSTM (samples, timesteps=1, features)
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
print(f"✓ Data reshaped to: {X_lstm.shape}")

# Predict
print("   Running model inference...")
predictions = model.predict(X_lstm, batch_size=32, verbose=0)

print()
print(f"✓ Predictions complete!")
print(f"   Shape: {predictions.shape}")
print(f"   Temperature anomaly range: [{predictions[:, 0].min():.3f}, {predictions[:, 0].max():.3f}]")
print(f"   Precipitation anomaly range: [{predictions[:, 1].min():.3f}, {predictions[:, 1].max():.3f}]")
print()

# ========================================
# STEP 4: Calculate Performance Metrics
# ========================================
print("📊 Performance Metrics:")
print("-"*80)

# Temperature anomaly metrics
temp_mae = mean_absolute_error(y[:, 0], predictions[:, 0])
temp_rmse = np.sqrt(mean_squared_error(y[:, 0], predictions[:, 0]))
temp_r2 = r2_score(y[:, 0], predictions[:, 0])

print(f"Temperature Anomaly:")
print(f"   MAE:  {temp_mae:.6f}")
print(f"   RMSE: {temp_rmse:.6f}")
print(f"   R²:   {temp_r2:.4f}")

# Precipitation anomaly metrics
precip_mae = mean_absolute_error(y[:, 1], predictions[:, 1])
precip_rmse = np.sqrt(mean_squared_error(y[:, 1], predictions[:, 1]))
precip_r2 = r2_score(y[:, 1], predictions[:, 1])

print()
print(f"Precipitation Anomaly:")
print(f"   MAE:  {precip_mae:.6f}")
print(f"   RMSE: {precip_rmse:.6f}")
print(f"   R²:   {precip_r2:.4f}")
print()

# ========================================
# STEP 5: Save Predictions
# ========================================
print("💾 Saving predictions...")
print("-"*80)

# Add predictions to dataframe
results_df = df_transformed.copy()
results_df['T2M_anom_predicted'] = predictions[:, 0]
results_df['PRECTOTCORR_anom_predicted'] = predictions[:, 1]

# Calculate prediction errors
results_df['T2M_error'] = results_df['T2M_anom'] - results_df['T2M_anom_predicted']
results_df['PRECTOTCORR_error'] = results_df['PRECTOTCORR_anom'] - results_df['PRECTOTCORR_anom_predicted']

# Save to CSV
results_path = RESULTS_DIR / 'climate_predictions_lstm_local.csv'
results_df.to_csv(results_path, index=False)
print(f"✓ Predictions saved to: {results_path}")
print()

# ========================================
# STEP 6: Visualize Results
# ========================================
print("📈 Creating visualizations...")
print("-"*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Temperature: Actual vs Predicted
axes[0, 0].scatter(y[:, 0], predictions[:, 0], alpha=0.3, s=20)
axes[0, 0].plot([y[:, 0].min(), y[:, 0].max()], 
                [y[:, 0].min(), y[:, 0].max()], 
                'r--', linewidth=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Temperature Anomaly', fontsize=12)
axes[0, 0].set_ylabel('Predicted Temperature Anomaly', fontsize=12)
axes[0, 0].set_title(f'Temperature Predictions (R²={temp_r2:.4f})', 
                     fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Precipitation: Actual vs Predicted
axes[0, 1].scatter(y[:, 1], predictions[:, 1], alpha=0.3, s=20, color='green')
axes[0, 1].plot([y[:, 1].min(), y[:, 1].max()], 
                [y[:, 1].min(), y[:, 1].max()], 
                'r--', linewidth=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('Actual Precipitation Anomaly', fontsize=12)
axes[0, 1].set_ylabel('Predicted Precipitation Anomaly', fontsize=12)
axes[0, 1].set_title(f'Precipitation Predictions (R²={precip_r2:.4f})', 
                     fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Temperature: Error Distribution
temp_errors = y[:, 0] - predictions[:, 0]
axes[1, 0].hist(temp_errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[1, 0].set_xlabel('Prediction Error', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title(f'Temperature Error Distribution (Mean={temp_errors.mean():.4f})', 
                     fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Precipitation: Error Distribution
precip_errors = y[:, 1] - predictions[:, 1]
axes[1, 1].hist(precip_errors, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
axes[1, 1].set_xlabel('Prediction Error', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title(f'Precipitation Error Distribution (Mean={precip_errors.mean():.4f})', 
                     fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

# Save visualization
viz_path = RESULTS_DIR / 'local_predictions_visualization.png'
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {viz_path}")
plt.close()

# ========================================
# STEP 7: Summary Statistics
# ========================================
print()
print("📊 Summary Statistics:")
print("-"*80)

print(f"Total predictions: {len(predictions):,}")
print()
print(f"Temperature Anomaly Predictions:")
print(f"   Mean: {predictions[:, 0].mean():.4f}")
print(f"   Std:  {predictions[:, 0].std():.4f}")
print(f"   Min:  {predictions[:, 0].min():.4f}")
print(f"   Max:  {predictions[:, 0].max():.4f}")
print()
print(f"Precipitation Anomaly Predictions:")
print(f"   Mean: {predictions[:, 1].mean():.4f}")
print(f"   Std:  {predictions[:, 1].std():.4f}")
print(f"   Min:  {predictions[:, 1].min():.4f}")
print(f"   Max:  {predictions[:, 1].max():.4f}")

# ========================================
# FINAL SUMMARY
# ========================================
print()
print("="*80)
print("✅ INFERENCE COMPLETE!")
print("="*80)
print()
print(f"Results saved to: {RESULTS_DIR}")
print(f"   1. {results_path.name} - Predictions CSV")
print(f"   2. {viz_path.name} - Visualization")
print()
print("="*80)
