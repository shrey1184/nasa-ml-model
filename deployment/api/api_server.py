"""
NASA Climate Model - Production API Server

FastAPI server for real-time climate anomaly predictions.
Fetches live data from NASA POWER API and returns predictions.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json
import requests
import logging
from tensorflow import keras

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NASA Climate Prediction API",
    description="Real-time climate anomaly predictions using LSTM model",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODEL_BUNDLE_DIR = Path(__file__).parent.parent / 'model_bundle'

# Global variables for model components
model = None
feature_scaler = None
cleaning_scalers = None
cleaning_encoders = None
config = None
metadata = None

# ==========================================
# Pydantic Models
# ==========================================

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude (-90 to 90)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude (-180 to 180)")
    months_back: int = Field(24, ge=12, le=60, description="Number of months to fetch (12-60)")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    location: Dict[str, float]
    data_fetched: bool
    samples: int
    predictions: List[Dict[str, Any]]
    summary: Dict[str, Any]
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str

# ==========================================
# Startup: Load Model Components
# ==========================================

@app.on_event("startup")
async def load_model_components():
    """Load all model components on startup"""
    global model, feature_scaler, cleaning_scalers, cleaning_encoders, config, metadata
    
    try:
        logger.info("Loading model components...")
        
        # Load LSTM model
        model_path = MODEL_BUNDLE_DIR / 'climate_lstm_model.keras'
        model = keras.models.load_model(str(model_path))
        logger.info(f"✓ Model loaded: {model_path.name}")
        
        # Load feature scaler
        scaler_path = MODEL_BUNDLE_DIR / 'lstm_scaler.pkl'
        feature_scaler = joblib.load(str(scaler_path))
        logger.info(f"✓ Feature scaler loaded: {scaler_path.name}")
        
        # Load cleaning components
        cleaning_scalers_path = MODEL_BUNDLE_DIR / 'cleaning_scalers.pkl'
        cleaning_scalers = joblib.load(str(cleaning_scalers_path))
        logger.info(f"✓ Cleaning scalers loaded: {cleaning_scalers_path.name}")
        
        cleaning_encoders_path = MODEL_BUNDLE_DIR / 'cleaning_encoders.pkl'
        cleaning_encoders = joblib.load(str(cleaning_encoders_path))
        logger.info(f"✓ Cleaning encoders loaded: {cleaning_encoders_path.name}")
        
        # Load configuration
        config_path = MODEL_BUNDLE_DIR / 'model_configuration.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"✓ Configuration loaded: {config_path.name}")
        
        # Load metadata
        metadata_path = MODEL_BUNDLE_DIR / 'deployment_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"✓ Metadata loaded: {metadata_path.name}")
        
        logger.info("All components loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model components: {str(e)}")
        raise

# ==========================================
# Helper Functions
# ==========================================

def fetch_nasa_power_data(latitude: float, longitude: float, months_back: int = 24) -> pd.DataFrame:
    """
    Fetch climate data from NASA POWER API
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        months_back: Number of months to fetch
        
    Returns:
        DataFrame with climate data
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months_back * 30)
        
        start_str = start_date.strftime("%Y%m")
        end_str = end_date.strftime("%Y%m")
        
        # NASA POWER API parameters
        params = [
            "T2M", "T2M_MAX", "T2M_MIN", 
            "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN"
        ]
        
        # Build API URL
        base_url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
        url = f"{base_url}?parameters={','.join(params)}&community=RE&longitude={longitude}&latitude={latitude}&start={start_str}&end={end_str}&format=JSON"
        
        logger.info(f"Fetching NASA data: lat={latitude}, lon={longitude}, {start_str}-{end_str}")
        
        # Make request
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract parameters
        parameters = data['properties']['parameter']
        
        # Create DataFrame
        records = []
        for date_str in parameters['T2M'].keys():
            if date_str == '13':  # Skip annual average
                continue
                
            record = {
                'date': date_str,
                'latitude': latitude,
                'longitude': longitude
            }
            
            for param in params:
                record[param] = parameters[param][date_str]
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Convert date to proper datetime
        df['date'] = pd.to_datetime(df['date'], format='%Y%m')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        logger.info(f"✓ Fetched {len(df)} months of data")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch NASA data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch NASA data: {str(e)}")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw NASA data to match training format
    
    Args:
        df: Raw NASA data
        
    Returns:
        Preprocessed DataFrame ready for model
    """
    try:
        # Calculate derived features
        df['T2M_range'] = df['T2M_MAX'] - df['T2M_MIN']
        df['heat_index'] = df['T2M'] + (0.5 * df['RH2M'] / 100)
        df['precip_log'] = np.log1p(df['PRECTOTCORR'])
        
        # Add temporal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['year_norm'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min() + 1e-8)
        
        # Add season (dummy encoding)
        seasons = pd.cut(df['month'], bins=[0, 3, 6, 9, 12], 
                        labels=['Winter', 'Spring', 'Summer', 'Fall'])
        season_dummies = pd.get_dummies(seasons, prefix='season')
        df = pd.concat([df, season_dummies], axis=1)
        
        # Ensure all season columns exist
        for season in ['season_Fall', 'season_Spring', 'season_Summer', 'season_Winter']:
            if season not in df.columns:
                df[season] = 0
        
        # Apply cleaning scalers to continuous features
        continuous_features = config['continuous_features']
        for feature in continuous_features:
            if feature in df.columns and feature in cleaning_scalers:
                df[f'{feature}_transformed'] = cleaning_scalers[feature].transform(df[[feature]])
            else:
                # If not in cleaning scalers, just use original
                df[f'{feature}_transformed'] = df[feature]
        
        # Calculate anomalies (using simple method - deviation from mean)
        for param in ['T2M', 'PRECTOTCORR']:
            param_mean = df[param].mean()
            df[f'{param}_anom'] = df[param] - param_mean
        
        logger.info(f"✓ Preprocessed data: {len(df)} samples")
        return df
        
    except Exception as e:
        logger.error(f"Failed to preprocess data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

def make_predictions(df: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using loaded model
    
    Args:
        df: Preprocessed data
        
    Returns:
        Predictions array
    """
    try:
        # Extract features in correct order
        feature_names = config['all_features_transformed']
        X = df[feature_names].values
        
        # Scale features
        X_scaled = feature_scaler.transform(X)
        
        # Reshape for LSTM (samples, timesteps=1, features)
        X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        # Predict
        predictions = model.predict(X_lstm, verbose=0)
        
        logger.info(f"✓ Generated {len(predictions)} predictions")
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ==========================================
# API Endpoints
# ==========================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "NASA Climate Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make predictions",
            "/model-info": "Model information"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="1.0.0"
    )

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": metadata['model_details']['architecture'],
        "input_features": metadata['model_details']['input_features'],
        "output_targets": metadata['model_details']['output_targets'],
        "performance": metadata['performance'],
        "features": {
            "required": config['all_features_transformed'],
            "continuous": config['continuous_features'],
            "temporal": config['temporal_features'],
            "spatial": config['spatial_features'],
            "engineered": config['engineered_features'],
            "categorical": config['categorical_features']
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make climate anomaly predictions for a location
    
    Args:
        request: Prediction request with latitude, longitude, and months_back
        
    Returns:
        Prediction response with anomaly predictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Fetch NASA data
        df = fetch_nasa_power_data(
            request.latitude, 
            request.longitude, 
            request.months_back
        )
        
        # Preprocess
        df = preprocess_data(df)
        
        # Make predictions
        predictions = make_predictions(df)
        
        # Format response
        prediction_list = []
        for i, (idx, row) in enumerate(df.iterrows()):
            prediction_list.append({
                "date": row['date'].strftime("%Y-%m"),
                "temperature_anomaly": float(predictions[i, 0]),
                "precipitation_anomaly": float(predictions[i, 1]),
                "actual_temperature": float(row['T2M']),
                "actual_precipitation": float(row['PRECTOTCORR'])
            })
        
        # Calculate summary statistics
        temp_predictions = predictions[:, 0]
        precip_predictions = predictions[:, 1]
        
        summary = {
            "temperature_anomaly": {
                "mean": float(np.mean(temp_predictions)),
                "std": float(np.std(temp_predictions)),
                "min": float(np.min(temp_predictions)),
                "max": float(np.max(temp_predictions))
            },
            "precipitation_anomaly": {
                "mean": float(np.mean(precip_predictions)),
                "std": float(np.std(precip_predictions)),
                "min": float(np.min(precip_predictions)),
                "max": float(np.max(precip_predictions))
            }
        }
        
        return PredictionResponse(
            location={
                "latitude": request.latitude,
                "longitude": request.longitude
            },
            data_fetched=True,
            samples=len(predictions),
            predictions=prediction_list,
            summary=summary,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Run Server
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    print("="*80)
    print("NASA CLIMATE PREDICTION API SERVER")
    print("="*80)
    print()
    print("Starting server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
