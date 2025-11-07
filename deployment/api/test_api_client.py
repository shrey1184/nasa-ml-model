#!/usr/bin/env python3
"""
API Test Client

Test the NASA Climate Prediction API with sample requests.
"""

import requests
import json
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

print("="*80)
print("NASA CLIMATE PREDICTION API - TEST CLIENT")
print("="*80)
print()

# ==========================================
# Test 1: Health Check
# ==========================================
print("Test 1: Health Check")
print("-"*80)

try:
    response = requests.get(f"{API_BASE_URL}/health")
    response.raise_for_status()
    data = response.json()
    
    print(f"Status: {data['status']}")
    print(f"Model Loaded: {data['model_loaded']}")
    print(f"Version: {data['version']}")
    print("✓ Health check passed")
except Exception as e:
    print(f"✗ Health check failed: {str(e)}")

print()

# ==========================================
# Test 2: Model Info
# ==========================================
print("Test 2: Model Information")
print("-"*80)

try:
    response = requests.get(f"{API_BASE_URL}/model-info")
    response.raise_for_status()
    data = response.json()
    
    print(f"Model Type: {data['model_type']}")
    print(f"Input Features: {data['input_features']}")
    print(f"Output Targets: {data['output_targets']}")
    print(f"R² Temperature: {data['performance']['r2_temperature']:.4f}")
    print(f"R² Precipitation: {data['performance']['r2_precipitation']:.4f}")
    print("✓ Model info retrieved")
except Exception as e:
    print(f"✗ Model info failed: {str(e)}")

print()

# ==========================================
# Test 3: Prediction Request
# ==========================================
print("Test 3: Make Prediction")
print("-"*80)

# Test locations
test_locations = [
    {"name": "New Delhi", "latitude": 28.6139, "longitude": 77.2090},
    {"name": "New York", "latitude": 40.7128, "longitude": -74.0060},
    {"name": "London", "latitude": 51.5074, "longitude": -0.1278},
]

for location in test_locations:
    print(f"\nLocation: {location['name']} ({location['latitude']}, {location['longitude']})")
    print("-"*40)
    
    try:
        # Make prediction request
        payload = {
            "latitude": location['latitude'],
            "longitude": location['longitude'],
            "months_back": 24
        }
        
        print("Sending request...")
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Request successful")
        print(f"  Samples: {data['samples']}")
        print(f"  Data fetched: {data['data_fetched']}")
        
        # Show summary
        summary = data['summary']
        print(f"\n  Temperature Anomaly:")
        print(f"    Mean: {summary['temperature_anomaly']['mean']:.4f}°C")
        print(f"    Range: [{summary['temperature_anomaly']['min']:.4f}, {summary['temperature_anomaly']['max']:.4f}]")
        
        print(f"\n  Precipitation Anomaly:")
        print(f"    Mean: {summary['precipitation_anomaly']['mean']:.4f} mm")
        print(f"    Range: [{summary['precipitation_anomaly']['min']:.4f}, {summary['precipitation_anomaly']['max']:.4f}]")
        
        # Show recent predictions
        print(f"\n  Recent Predictions (last 3 months):")
        for pred in data['predictions'][-3:]:
            print(f"    {pred['date']}: Temp={pred['temperature_anomaly']:>7.3f}°C, Precip={pred['precipitation_anomaly']:>7.3f}mm")
        
    except requests.exceptions.Timeout:
        print(f"✗ Request timeout (NASA API might be slow)")
    except Exception as e:
        print(f"✗ Prediction failed: {str(e)}")

print()
print("="*80)
print("Testing complete!")
print("="*80)
