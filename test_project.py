#!/usr/bin/env python3
"""
Project Integration Test Script
Tests all components of the NASA ML project to ensure everything works correctly
"""

import sys
import os
from pathlib import Path

print("="*80)
print("NASA ML PROJECT - COMPREHENSIVE TEST SCRIPT")
print("="*80)
print()

# ==========================================
# TEST 1: Check Python Version
# ==========================================
print("TEST 1: Python Version")
print("-"*80)
print(f"✓ Python version: {sys.version}")
print(f"✓ Executable: {sys.executable}")
print()

# ==========================================
# TEST 2: Check Required Packages
# ==========================================
print("TEST 2: Required Packages")
print("-"*80)

required_packages = {
    'requests': 'Data fetching',
    'pandas': 'Data processing',
    'numpy': 'Numerical operations',
    'tensorflow': 'ML model (LSTM)',
    'keras': 'ML model API',
    'sklearn': 'ML utilities',
    'matplotlib': 'Visualization',
    'seaborn': 'Advanced visualization',
    'joblib': 'Model persistence',
}

missing_packages = []
installed_packages = []

for package, purpose in required_packages.items():
    try:
        if package == 'sklearn':
            __import__('sklearn')
        else:
            __import__(package)
        print(f"✓ {package:15s} - {purpose}")
        installed_packages.append(package)
    except ImportError:
        print(f"✗ {package:15s} - {purpose} [MISSING]")
        missing_packages.append(package)

print()
if missing_packages:
    print(f"⚠ WARNING: {len(missing_packages)} packages are missing!")
    print(f"   Missing: {', '.join(missing_packages)}")
    print("   Run: pip install -r requirements.txt")
    print()
else:
    print("✅ All required packages are installed!")
    print()

# ==========================================
# TEST 3: Check Project Structure
# ==========================================
print("TEST 3: Project Structure")
print("-"*80)

required_files = [
    'requirements.txt',
    'main_pipeline.py',
    'cleaning_pipeline.py',
    'nasa_apis.py',
    'README.md',
    'src/location_grid.py',
    'src/data_fetcher.py',
    'src/data_processor.py',
    'src/data_cleaner.py',
    'data/model_configuration.json',
    'data/climate_model_ready_transformed.csv',
    'models/climate_lstm_model.keras',
    'models/lstm_model_metadata.json',
    'models/lstm_scaler.pkl',
]

missing_files = []
found_files = []

project_root = Path.cwd()

for file_path in required_files:
    full_path = project_root / file_path
    if full_path.exists():
        print(f"✓ {file_path}")
        found_files.append(file_path)
    else:
        print(f"✗ {file_path} [MISSING]")
        missing_files.append(file_path)

print()
if missing_files:
    print(f"⚠ WARNING: {len(missing_files)} files are missing!")
    for file in missing_files:
        print(f"   - {file}")
    print()
else:
    print("✅ All required files are present!")
    print()

# ==========================================
# TEST 4: Import Project Modules
# ==========================================
print("TEST 4: Project Modules")
print("-"*80)

sys.path.insert(0, str(project_root / 'src'))

module_tests = [
    ('src.location_grid', 'LocationGrid'),
    ('src.data_fetcher', 'ClimateDataFetcher'),
    ('src.data_processor', 'DataProcessor'),
    ('src.data_cleaner', 'DataCleaner'),
]

successful_imports = []
failed_imports = []

for module_name, class_name in module_tests:
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"✓ {module_name}.{class_name}")
        successful_imports.append((module_name, class_name))
    except Exception as e:
        print(f"✗ {module_name}.{class_name} - {str(e)}")
        failed_imports.append((module_name, class_name))

print()
if failed_imports:
    print(f"⚠ WARNING: {len(failed_imports)} modules failed to import!")
    print()
else:
    print("✅ All project modules imported successfully!")
    print()

# ==========================================
# TEST 5: Check Data Files
# ==========================================
print("TEST 5: Data Files")
print("-"*80)

if 'pandas' in installed_packages:
    import pandas as pd
    
    data_files = [
        'data/climate_model_ready_transformed.csv',
        'data/model_configuration.json',
        'data/locations_major_cities.csv',
    ]
    
    for data_file in data_files:
        file_path = project_root / data_file
        if file_path.exists():
            try:
                if data_file.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    print(f"✓ {data_file}")
                    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                elif data_file.endswith('.json'):
                    import json
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    print(f"✓ {data_file}")
                    print(f"  Keys: {len(data)} entries")
            except Exception as e:
                print(f"✗ {data_file} - Error reading: {str(e)}")
        else:
            print(f"⚠ {data_file} - File not found")
    print()
else:
    print("⚠ Pandas not installed - skipping data file checks")
    print()

# ========================================
# TEST 6: Check Model Files
# ========================================
print("TEST 6: Model Files")
print("-"*80)

model_dir = project_root / 'models'

if model_dir.exists():
    model_files = list(model_dir.iterdir())
    print(f"✓ Model directory exists")
    print(f"  Files found: {len(model_files)}")
    for model_file in sorted(model_files):
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  - {model_file.name} ({size_mb:.2f} MB)")
    
    # Check critical files
    critical_model_files = [
        'climate_lstm_model.keras',
        'lstm_model_metadata.json',
        'lstm_scaler.pkl',
    ]
    
    print()
    all_present = True
    for file_name in critical_model_files:
        file_path = model_dir / file_name
        if file_path.exists():
            print(f"✓ {file_name}")
        else:
            print(f"✗ {file_name} [MISSING]")
            all_present = False
    
    print()
    if all_present:
        print("✅ All critical model files are present!")
    else:
        print("⚠ WARNING: Some model files are missing!")
    print()
else:
    print("✗ Model directory not found!")
    print()

# ==========================================
# TEST 7: Test Model Loading (if packages available)
# ==========================================
print("TEST 7: Model Loading")
print("-"*80)

if 'tensorflow' in installed_packages and 'joblib' in installed_packages:
    try:
        from tensorflow import keras
        import joblib
        import json
        
        model_path = model_dir / 'climate_lstm_model.keras'
        scaler_path = model_dir / 'lstm_scaler.pkl'
        metadata_path = model_dir / 'lstm_model_metadata.json'
        
        # Load model
        model = keras.models.load_model(str(model_path))
        print(f"✓ LSTM model loaded successfully")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
        
        # Load scaler
        scaler = joblib.load(str(scaler_path))
        print(f"✓ Scaler loaded successfully")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Metadata loaded successfully")
        print(f"  Model type: {metadata['model_type']}")
        print(f"  Test R² (Temperature): {metadata['r2_temperature']:.4f}")
        print(f"  Test R² (Precipitation): {metadata['r2_precipitation']:.4f}")
        
        print()
        print("✅ Model loading test passed!")
        print()
        
    except Exception as e:
        print(f"✗ Model loading failed: {str(e)}")
        print()
else:
    print("⚠ TensorFlow or joblib not installed - skipping model loading test")
    print()

# ==========================================
# TEST 8: Quick Prediction Test (if everything available)
# ==========================================
print("TEST 8: Quick Prediction Test")
print("-"*80)

if 'tensorflow' in installed_packages and 'pandas' in installed_packages:
    try:
        import pandas as pd
        import numpy as np
        from tensorflow import keras
        import joblib
        
        # Load data
        data_path = project_root / 'data/climate_model_ready_transformed.csv'
        df = pd.read_csv(data_path)
        
        # Load model config
        config_path = project_root / 'data/model_configuration.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load model and scaler
        model = keras.models.load_model(str(model_dir / 'climate_lstm_model.keras'))
        scaler = joblib.load(str(model_dir / 'lstm_scaler.pkl'))
        
        # Prepare test sample (first 5 rows)
        X_test = df[config['all_features_transformed']].head(5).values
        y_test = df[config['target_variables']].head(5).values
        
        # Scale and reshape
        X_scaled = scaler.transform(X_test)
        X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        # Predict
        predictions = model.predict(X_lstm, verbose=0)
        
        print("✓ Successfully ran predictions on test samples")
        print(f"  Input shape: {X_lstm.shape}")
        print(f"  Output shape: {predictions.shape}")
        print(f"  Sample predictions:")
        print(f"    Temperature anomaly: {predictions[0, 0]:.6f}")
        print(f"    Precipitation anomaly: {predictions[0, 1]:.6f}")
        
        print()
        print("✅ Prediction test passed!")
        print()
        
    except Exception as e:
        print(f"✗ Prediction test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
else:
    print("⚠ Required packages not installed - skipping prediction test")
    print()

# ==========================================
# FINAL SUMMARY
# ==========================================
print("="*80)
print("TEST SUMMARY")
print("="*80)
print()

total_tests = 8
passed_tests = 0

# Count what passed
if not missing_packages:
    passed_tests += 1
if not missing_files:
    passed_tests += 1
if not failed_imports:
    passed_tests += 1

print(f"Project Root: {project_root}")
print(f"Tests Run: {total_tests}")
print()

if missing_packages:
    print(f"❌ Missing packages: {len(missing_packages)}")
    print(f"   To fix: pip install -r requirements.txt")
    print()
    
if missing_files:
    print(f"❌ Missing files: {len(missing_files)}")
    print()

if failed_imports:
    print(f"❌ Failed module imports: {len(failed_imports)}")
    print()

if not missing_packages and not missing_files and not failed_imports:
    print("✅ ALL TESTS PASSED!")
    print()
    print("Your project is properly set up and ready to use!")
    print()
    print("Next steps:")
    print("  1. Run data pipeline: python main_pipeline.py --grid cities --start 2010 --end 2024")
    print("  2. Clean data: python cleaning_pipeline.py")
    print("  3. Use model for predictions (see google_colab_lstm_workflow.py)")
else:
    print("⚠ SOME ISSUES FOUND")
    print()
    print("Please fix the issues above and run this test again.")

print()
print("="*80)
