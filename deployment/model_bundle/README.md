# NASA Climate Model - Deployment Bundle

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
