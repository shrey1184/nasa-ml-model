#!/usr/bin/env python3
"""
Export Model Bundle for Production Deployment

This script creates a complete deployment package including:
1. Trained LSTM model
2. Feature scaler
3. Data cleaning components
4. Model configuration
5. Deployment metadata
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
import joblib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'
DEPLOYMENT_DIR = PROJECT_ROOT / 'deployment'
EXPORT_DIR = DEPLOYMENT_DIR / 'model_bundle'

print("="*80)
print("NASA CLIMATE MODEL - DEPLOYMENT EXPORT")
print("="*80)
print()

# Create export directory
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
print(f"📁 Export directory: {EXPORT_DIR}")
print()

# ==========================================
# Step 1: Copy Model Files
# ==========================================
print("Step 1: Copying Model Files")
print("-"*80)

files_to_copy = {
    'climate_lstm_model.keras': 'Main LSTM model',
    'lstm_scaler.pkl': 'Feature scaler (StandardScaler)',
    'lstm_model_metadata.json': 'Model performance metrics',
    'cleaning_scalers.pkl': 'Data cleaning scalers',
    'cleaning_encoders.pkl': 'Categorical encoders',
}

copied_files = []
missing_files = []

for filename, description in files_to_copy.items():
    source = MODELS_DIR / filename
    dest = EXPORT_DIR / filename
    
    if source.exists():
        shutil.copy2(source, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"✓ {filename:30s} ({size_mb:>6.2f} MB) - {description}")
        copied_files.append(filename)
    else:
        print(f"✗ {filename:30s} [MISSING] - {description}")
        missing_files.append(filename)

print()

# ==========================================
# Step 2: Copy Configuration
# ==========================================
print("Step 2: Copying Configuration Files")
print("-"*80)

config_source = DATA_DIR / 'model_configuration.json'
config_dest = EXPORT_DIR / 'model_configuration.json'

if config_source.exists():
    shutil.copy2(config_source, config_dest)
    print(f"✓ model_configuration.json")
    copied_files.append('model_configuration.json')
else:
    print(f"✗ model_configuration.json [MISSING]")
    missing_files.append('model_configuration.json')

print()

# ==========================================
# Step 3: Create Deployment Metadata
# ==========================================
print("Step 3: Creating Deployment Metadata")
print("-"*80)

# Load model metadata
with open(MODELS_DIR / 'lstm_model_metadata.json', 'r') as f:
    model_metadata = json.load(f)

# Load model configuration
with open(DATA_DIR / 'model_configuration.json', 'r') as f:
    model_config = json.load(f)

deployment_metadata = {
    "deployment_info": {
        "export_date": datetime.now().isoformat(),
        "version": "1.0.0",
        "model_type": "LSTM",
        "framework": "TensorFlow/Keras",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    },
    "model_details": {
        "architecture": "LSTM",
        "input_features": model_metadata['n_features'],
        "output_targets": model_metadata['n_targets'],
        "input_shape": model_metadata['input_shape'],
        "output_shape": model_metadata['output_shape'],
        "training_samples": model_metadata['train_samples'],
        "validation_samples": model_metadata['val_samples'],
        "test_samples": model_metadata['test_samples']
    },
    "performance": {
        "test_loss": model_metadata['test_loss'],
        "test_mae": model_metadata['test_mae'],
        "test_rmse": model_metadata['test_rmse'],
        "r2_temperature": model_metadata['r2_temperature'],
        "r2_precipitation": model_metadata['r2_precipitation']
    },
    "features": {
        "input_features": model_config['all_features_transformed'],
        "target_variables": model_config['target_variables'],
        "continuous_features": model_config['continuous_features'],
        "temporal_features": model_config['temporal_features'],
        "spatial_features": model_config['spatial_features'],
        "engineered_features": model_config['engineered_features'],
        "categorical_features": model_config['categorical_features']
    },
    "preprocessing": {
        "scaler": "StandardScaler",
        "scaler_file": "lstm_scaler.pkl",
        "cleaning_scalers_file": "cleaning_scalers.pkl",
        "cleaning_encoders_file": "cleaning_encoders.pkl"
    },
    "api_requirements": {
        "nasa_power_api": "https://power.larc.nasa.gov/api/temporal/monthly/point",
        "required_parameters": [
            "T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", 
            "RH2M", "ALLSKY_SFC_SW_DWN"
        ],
        "temporal_range": "Last 24 months minimum",
        "data_format": "Monthly time series"
    },
    "prediction_outputs": {
        "T2M_anom": "Temperature anomaly (°C)",
        "PRECTOTCORR_anom": "Precipitation anomaly (mm)"
    },
    "files": {
        "model": "climate_lstm_model.keras",
        "feature_scaler": "lstm_scaler.pkl",
        "cleaning_scalers": "cleaning_scalers.pkl",
        "cleaning_encoders": "cleaning_encoders.pkl",
        "configuration": "model_configuration.json",
        "metadata": "deployment_metadata.json"
    }
}

metadata_path = EXPORT_DIR / 'deployment_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(deployment_metadata, f, indent=2)

print(f"✓ deployment_metadata.json created")
print()

# ==========================================
# Step 4: Create Requirements File
# ==========================================
print("Step 4: Creating Deployment Requirements")
print("-"*80)

deployment_requirements = """# NASA Climate Model - Deployment Requirements
# Minimal dependencies for production deployment

# Core ML/Data
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0

# API Framework
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0

# HTTP Requests (for NASA API)
requests>=2.31.0
python-dateutil>=2.8.2

# Optional: Logging and monitoring
python-json-logger>=2.0.0
"""

req_path = EXPORT_DIR / 'requirements_deployment.txt'
with open(req_path, 'w') as f:
    f.write(deployment_requirements)

print(f"✓ requirements_deployment.txt created")
print()

# ==========================================
# Step 5: Create README
# ==========================================
print("Step 5: Creating Deployment README")
print("-"*80)

readme_content = """# NASA Climate Model - Deployment Bundle

## 📦 Package Contents

This bundle contains everything needed to deploy the NASA Climate LSTM model in production.

### Files Included

1. **climate_lstm_model.keras** (3.79 MB)
   - Trained LSTM model for climate anomaly prediction
   - Input: 18 features
   - Output: 2 targets (temperature & precipitation anomalies)

2. **lstm_scaler.pkl** (1 KB)
   - StandardScaler for model input features
   - Fitted on training data

3. **cleaning_scalers.pkl** (1 KB)
   - Scalers used during data cleaning
   - Applied to raw NASA data before model input

4. **cleaning_encoders.pkl** (< 1 KB)
   - Encoders for categorical variables (season, etc.)

5. **model_configuration.json**
   - Feature definitions and model settings
   - Lists all required input features

6. **deployment_metadata.json**
   - Model performance metrics
   - API requirements
   - Preprocessing instructions

7. **requirements_deployment.txt**
   - Python dependencies for deployment

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements_deployment.txt
```

### Load Model

```python
from tensorflow import keras
import joblib
import json

# Load model
model = keras.models.load_model('climate_lstm_model.keras')

# Load scalers
feature_scaler = joblib.load('lstm_scaler.pkl')
cleaning_scalers = joblib.load('cleaning_scalers.pkl')
cleaning_encoders = joblib.load('cleaning_encoders.pkl')

# Load configuration
with open('model_configuration.json', 'r') as f:
    config = json.load(f)
```

## 🔄 Prediction Workflow

### 1. Fetch NASA POWER Data

Fetch climate data for target location:
- Last 24 months of data
- Parameters: T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, RH2M, ALLSKY_SFC_SW_DWN

### 2. Clean & Preprocess

```python
# Apply cleaning scalers to raw data
# Encode categorical variables (season)
# Calculate derived features (T2M_range, heat_index, precip_log)
```

### 3. Transform Features

```python
# Extract features in correct order
X = data[config['all_features_transformed']].values

# Scale using feature_scaler
X_scaled = feature_scaler.transform(X)

# Reshape for LSTM
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
```

### 4. Predict

```python
predictions = model.predict(X_lstm)
# predictions[:, 0] = Temperature anomaly
# predictions[:, 1] = Precipitation anomaly
```

## 📊 Model Performance

- **Temperature Anomaly**: R² = 0.318, RMSE = 0.216
- **Precipitation Anomaly**: R² = 0.739, RMSE = 0.237

## 🌐 API Deployment

See `api/` folder for FastAPI implementation.

## 📝 Notes

- Model expects exactly 18 input features in specific order
- All features must be scaled using provided scalers
- Input shape: (batch_size, 1, 18)
- Output shape: (batch_size, 2)

## 📞 Support

For issues or questions, refer to the main project repository.
"""

readme_path = EXPORT_DIR / 'README.md'
with open(readme_path, 'w') as f:
    f.write(readme_content)

print(f"✓ README.md created")
print()

# ==========================================
# Summary
# ==========================================
print("="*80)
print("EXPORT SUMMARY")
print("="*80)
print()

total_size = sum(f.stat().st_size for f in EXPORT_DIR.iterdir() if f.is_file())
total_size_mb = total_size / (1024 * 1024)

print(f"✅ Export completed successfully!")
print()
print(f"Location: {EXPORT_DIR}")
print(f"Total files: {len(list(EXPORT_DIR.iterdir()))}")
print(f"Total size: {total_size_mb:.2f} MB")
print()

if copied_files:
    print(f"Copied files ({len(copied_files)}):")
    for f in copied_files:
        print(f"  ✓ {f}")
    print()

if missing_files:
    print(f"⚠️  Missing files ({len(missing_files)}):")
    for f in missing_files:
        print(f"  ✗ {f}")
    print()

print("Next steps:")
print("  1. Review deployment bundle in: deployment/model_bundle/")
print("  2. Test API server: python deployment/api/api_server.py")
print("  3. Deploy to production server")
print()
print("="*80)
