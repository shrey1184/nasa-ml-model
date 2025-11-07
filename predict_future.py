#!/usr/bin/env python3
"""
NASA Climate Future Prediction Tool
Make climate trend predictions for future dates and locations using trained LSTM model
"""

import numpy as np
import pandas as pd
from tensorflow import keras
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'

class ClimateFuturePredictor:
    def __init__(self):
        """Initialize the climate predictor with trained model and scalers"""
        print("🌍 NASA Climate Future Predictor")
        print("=" * 50)
        
        # Load model components
        self.model = keras.models.load_model(str(MODEL_DIR / 'climate_lstm_model.keras'))
        self.scaler = joblib.load(str(MODEL_DIR / 'lstm_scaler.pkl'))
        self.cleaning_scalers = joblib.load(str(MODEL_DIR / 'cleaning_scalers.pkl'))
        
        # Load configuration
        with open(DATA_DIR / 'model_configuration.json', 'r') as f:
            self.config = json.load(f)
        
        # Load baseline climatology (calculated from training data)
        baseline_file = DATA_DIR / 'climate_baselines.json'
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                self.baselines = json.load(f)
            print("✓ Loaded climatological baselines from training data")
        else:
            print("⚠️  Warning: Baseline climatology not found, will use estimates")
            self.baselines = None
        
        print("✓ Model and scalers loaded successfully")
        print(f"✓ Model expects {self.config['total_features']} features")
        print()
    
    def create_future_features(self, latitude, longitude, target_year, target_month):
        """
        Create feature vector for future prediction
        
        Args:
            latitude (float): Target latitude (-90 to 90)
            longitude (float): Target longitude (-180 to 180)
            target_year (int): Future year to predict (e.g., 2024, 2025, 2030)
            target_month (int): Month to predict (1-12)
        
        Returns:
            np.array: Feature vector ready for model prediction
        """
        
        # Create base feature dictionary
        features = {}
        
        # Spatial features
        features['latitude'] = latitude
        features['longitude'] = longitude
        
        # Temporal features
        features['month_sin'] = np.sin(2 * np.pi * target_month / 12)
        features['month_cos'] = np.cos(2 * np.pi * target_month / 12)
        
        # Normalize year based on training data range (assume 2010-2024 was training range)
        # You can adjust this based on your actual training data range
        training_year_min = 2010
        training_year_max = 2024
        features['year_norm'] = (target_year - training_year_min) / (training_year_max - training_year_min)
        
        # For future predictions, we need to estimate climate parameters
        # Using climatological averages based on location and month
        features.update(self._estimate_climate_params(latitude, longitude, target_month))
        
        # Calculate engineered features
        features['T2M_range'] = features['T2M_MAX'] - features['T2M_MIN']
        features['heat_index'] = features['T2M'] + (0.5 * features['RH2M'] / 100)
        features['precip_log'] = np.log1p(features['PRECTOTCORR'])
        
        # Season dummy variables
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Fall', 10: 'Fall', 11: 'Fall'}
        
        current_season = season_map[target_month]
        features['season_Fall'] = 1 if current_season == 'Fall' else 0
        features['season_Spring'] = 1 if current_season == 'Spring' else 0
        features['season_Summer'] = 1 if current_season == 'Summer' else 0
        features['season_Winter'] = 1 if current_season == 'Winter' else 0
        
        # Apply transformations to continuous features
        continuous_features = self.config['continuous_features']
        for feature in continuous_features:
            if feature in self.cleaning_scalers and feature in features:
                # Apply the same transformation used in training
                feature_value = np.array([[features[feature]]])
                features[f'{feature}_transformed'] = self.cleaning_scalers[feature].transform(feature_value)[0, 0]
            else:
                # If no scaler available, use original value
                features[f'{feature}_transformed'] = features[feature]
        
        # Create feature vector in correct order
        feature_vector = []
        for feature_name in self.config['all_features_transformed']:
            feature_vector.append(features[feature_name])
        
        return np.array(feature_vector).reshape(1, -1)
    
    def _estimate_climate_params(self, latitude, longitude, month):
        """
        Estimate climatological parameters based on location and month
        Uses real baseline climatology from training data when available
        """
        
        # Try to find exact baseline match from training data
        if self.baselines is not None:
            # Round coordinates to match baseline data
            lat_rounded = round(latitude, 2)
            lon_rounded = round(longitude, 2)
            key = f"{lat_rounded:.2f},{lon_rounded:.2f},{month}"
            
            if key in self.baselines:
                # Use real baseline from training data
                baseline = self.baselines[key]
                T2M = baseline['temperature']
                PRECTOTCORR = baseline['precipitation']
                
                # Estimate other parameters based on temperature and precipitation
                T2M_MAX = T2M + 7
                T2M_MIN = T2M - 7
                
                # Humidity (temperature and precipitation dependent)
                RH2M = 70 - T2M * 0.8 + (PRECTOTCORR / 100) * 10
                RH2M = np.clip(RH2M, 20, 95)
                
                # Solar radiation (latitude and season dependent)
                abs_lat = abs(latitude)
                season_factor = np.sin(2 * np.pi * (month - 1) / 12 - np.pi / 2)
                lat_factor = 1 - (abs_lat / 90) * 0.3
                solar_season = 0.7 + 0.3 * season_factor
                ALLSKY_SFC_SW_DWN = 220 * lat_factor * solar_season
                
                # Specific humidity (approximate from RH and T)
                QV2M = 0.622 * 0.01 * RH2M * np.exp(17.27 * T2M / (T2M + 237.3)) / 1013.25
                QV2M = QV2M * 15  # Scale to match training data range
                
                # Store baseline for later use
                self._baseline_temp = T2M
                self._baseline_precip = PRECTOTCORR
                
                return {
                    'T2M': T2M,
                    'T2M_MAX': T2M_MAX,
                    'T2M_MIN': T2M_MIN,
                    'PRECTOTCORR': PRECTOTCORR,
                    'RH2M': RH2M,
                    'ALLSKY_SFC_SW_DWN': ALLSKY_SFC_SW_DWN,
                    'QV2M': QV2M
                }
        
        # Fall back to estimation if no baseline found
        print(f"⚠️  No baseline found for ({latitude:.2f}, {longitude:.2f}), month {month} - using estimates")
        
        # Determine hemisphere and climate zone
        abs_lat = abs(latitude)
        is_northern = latitude >= 0
        
        # Month adjustment for southern hemisphere (opposite seasons)
        effective_month = month if is_northern else ((month + 6 - 1) % 12) + 1
        
        # Seasonal temperature variation (normalized -1 to 1)
        # Peak in July (NH) or January (SH), minimum in January (NH) or July (SH)
        season_factor = np.sin(2 * np.pi * (effective_month - 1) / 12 - np.pi / 2)
        
        # Temperature estimation based on latitude zones
        if abs_lat < 23.5:  # Tropical
            base_temp = 27
            seasonal_range = 3
        elif abs_lat < 35:  # Subtropical
            base_temp = 22
            seasonal_range = 10
        elif abs_lat < 50:  # Temperate
            base_temp = 15
            seasonal_range = 15
        elif abs_lat < 60:  # Cold temperate
            base_temp = 8
            seasonal_range = 20
        else:  # Polar
            base_temp = -5
            seasonal_range = 25
        
        # Calculate temperature with seasonal variation
        T2M = base_temp + season_factor * seasonal_range
        T2M_MAX = T2M + 7
        T2M_MIN = T2M - 7
        
        # Precipitation estimation (monsoon/seasonal patterns)
        if abs_lat < 10:  # Equatorial (high rainfall year-round)
            base_precip = 200
            seasonal_factor = 1.0
        elif abs_lat < 30:  # Monsoon regions
            base_precip = 100
            # Summer monsoon in tropics/subtropics
            monsoon_peak = np.sin(2 * np.pi * (effective_month - 6) / 12)
            seasonal_factor = 0.5 + 1.5 * max(0, monsoon_peak)
        elif abs_lat < 50:  # Mid-latitudes
            base_precip = 60
            # More uniform, slight summer peak
            seasonal_factor = 0.8 + 0.4 * np.sin(2 * np.pi * (effective_month - 6) / 12)
        else:  # High latitudes
            base_precip = 40
            seasonal_factor = 1.0
        
        PRECTOTCORR = base_precip * seasonal_factor
        
        # Humidity (temperature and precipitation dependent)
        RH2M = 70 - T2M * 0.8 + (PRECTOTCORR / 100) * 10
        RH2M = np.clip(RH2M, 20, 95)
        
        # Solar radiation (latitude and season dependent)
        # Maximum at equator, varies with season at higher latitudes
        lat_factor = 1 - (abs_lat / 90) * 0.3
        solar_season = 0.7 + 0.3 * season_factor
        ALLSKY_SFC_SW_DWN = 220 * lat_factor * solar_season
        
        # Store baseline for later use
        self._baseline_temp = T2M
        self._baseline_precip = PRECTOTCORR
        
        # Specific humidity (approximate from RH and T)
        QV2M = 0.622 * 0.01 * RH2M * np.exp(17.27 * T2M / (T2M + 237.3)) / 1013.25
        QV2M = QV2M * 15  # Scale to match training data range
        
        return {
            'T2M': T2M,
            'T2M_MAX': T2M_MAX,
            'T2M_MIN': T2M_MIN,
            'PRECTOTCORR': PRECTOTCORR,
            'RH2M': RH2M,
            'ALLSKY_SFC_SW_DWN': ALLSKY_SFC_SW_DWN,
            'QV2M': QV2M
        }
    
    def predict_future_climate(self, latitude, longitude, target_year, target_month):
        """
        Predict climate anomalies for a future date and location
        
        Args:
            latitude (float): Target latitude
            longitude (float): Target longitude  
            target_year (int): Future year
            target_month (int): Target month (1-12)
        
        Returns:
            dict: Prediction results
        """
        
        # Validate inputs
        if not (-90 <= latitude <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if not (-180 <= longitude <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        if not (1 <= target_month <= 12):
            raise ValueError("Month must be between 1 and 12")
        
        print(f"🎯 Predicting climate for:")
        print(f"   📍 Location: {latitude:.4f}°, {longitude:.4f}°")
        print(f"   📅 Date: {target_year}-{target_month:02d}")
        
        # Create feature vector
        features = self.create_future_features(latitude, longitude, target_year, target_month)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Reshape for LSTM (samples, timesteps=1, features)
        features_lstm = features_scaled.reshape((1, 1, features_scaled.shape[1]))
        
        # Make prediction
        prediction = self.model.predict(features_lstm, verbose=0)
        
        # Extract results
        temp_anomaly = prediction[0, 0]
        precip_anomaly = prediction[0, 1]
        
        # Calculate actual predicted values (baseline + anomaly)
        predicted_temp = self._baseline_temp + temp_anomaly
        predicted_precip = max(0, self._baseline_precip + precip_anomaly)  # Can't be negative
        
        results = {
            'location': {
                'latitude': latitude,
                'longitude': longitude
            },
            'date': {
                'year': target_year,
                'month': target_month,
                'month_name': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][target_month - 1]
            },
            'baseline': {
                'temperature': float(self._baseline_temp),
                'precipitation': float(self._baseline_precip)
            },
            'anomalies': {
                'temperature': float(temp_anomaly),
                'precipitation': float(precip_anomaly)
            },
            'predictions': {
                'temperature': float(predicted_temp),
                'precipitation': float(predicted_precip)
            },
            'interpretation': {
                'temperature': 'warmer than average' if temp_anomaly > 0 else 'cooler than average',
                'precipitation': 'wetter than average' if precip_anomaly > 0 else 'drier than average'
            }
        }
        
        return results
    
    def predict_multiple_months(self, latitude, longitude, start_year, start_month, num_months):
        """
        Predict climate for multiple consecutive months
        
        Args:
            latitude, longitude: Location
            start_year, start_month: Starting date
            num_months: Number of months to predict
        
        Returns:
            list: List of prediction results
        """
        
        results = []
        current_year = start_year
        current_month = start_month
        
        for i in range(num_months):
            result = self.predict_future_climate(latitude, longitude, current_year, current_month)
            results.append(result)
            
            # Advance to next month
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
        
        return results


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='NASA Climate Future Predictor')
    parser.add_argument('--lat', type=float, required=True, help='Latitude (-90 to 90)')
    parser.add_argument('--lon', type=float, required=True, help='Longitude (-180 to 180)')
    parser.add_argument('--year', type=int, required=True, help='Target year (e.g., 2025, 2030)')
    parser.add_argument('--month', type=int, required=True, help='Target month (1-12)')
    parser.add_argument('--months', type=int, default=1, help='Number of consecutive months to predict')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = ClimateFuturePredictor()
        
        if args.months == 1:
            # Single month prediction
            result = predictor.predict_future_climate(args.lat, args.lon, args.year, args.month)
            
            print("\n🌡️ PREDICTION RESULTS")
            print("=" * 50)
            print(f"📍 Location: {result['location']['latitude']:.4f}°, {result['location']['longitude']:.4f}°")
            print(f"📅 Date: {result['date']['year']}-{result['date']['month']:02d} ({result['date']['month_name']})")
            print()
            print(f"📊 Baseline (Historical Average):")
            print(f"   Temperature: {result['baseline']['temperature']:.2f}°C")
            print(f"   Precipitation: {result['baseline']['precipitation']:.2f}mm")
            print()
            print(f"🔮 Predicted Values:")
            print(f"   🌡️  Temperature: {result['predictions']['temperature']:.2f}°C")
            print(f"   🌧️  Precipitation: {result['predictions']['precipitation']:.2f}mm")
            print()
            print(f"📈 Anomalies (Difference from Average):")
            print(f"   Temperature: {result['anomalies']['temperature']:+.3f}°C ({result['interpretation']['temperature']})")
            print(f"   Precipitation: {result['anomalies']['precipitation']:+.3f}mm ({result['interpretation']['precipitation']})")
            
        else:
            # Multiple months prediction
            results = predictor.predict_multiple_months(args.lat, args.lon, args.year, args.month, args.months)
            
            print(f"\n🌡️ PREDICTION RESULTS ({args.months} months)")
            print("=" * 80)
            print(f"📍 Location: {args.lat:.4f}°, {args.lon:.4f}°")
            print()
            print("Date        | Predicted Temp | Predicted Precip | Baseline Temp | Baseline Precip")
            print("-" * 80)
            
            for result in results:
                date_str = f"{result['date']['year']}-{result['date']['month']:02d}"
                pred_temp = f"{result['predictions']['temperature']:6.2f}°C"
                pred_precip = f"{result['predictions']['precipitation']:7.2f}mm"
                base_temp = f"{result['baseline']['temperature']:6.2f}°C"
                base_precip = f"{result['baseline']['precipitation']:7.2f}mm"
                print(f"{date_str} | {pred_temp} | {pred_precip} | {base_temp} | {base_precip}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())