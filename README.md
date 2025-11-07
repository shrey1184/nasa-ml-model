# NASA Climate Trends Prediction - ML Project

Complete pipeline for collecting and processing NASA POWER climate data for ML model training.

## 🚀 Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Make predictions with trained model
python local_inference.py

# Fetch new climate data
python main_pipeline.py --grid cities --start 2010 --end 2024

# Clean and process data
python cleaning_pipeline.py

# Run tests
python test_project.py
```

## 🤖 Model Performance

**LSTM Model** for climate anomaly prediction:
- **Temperature Anomaly**: R² = 0.35, RMSE = 0.107
- **Precipitation Anomaly**: R² = 0.79, RMSE = 0.237
- **Architecture**: LSTM with 18 input features, 2 outputs
- **Training**: 1,260 samples, 100 epochs

## 📁 Project Structure

```
ml nasa/
├── main_pipeline.py          # Data collection pipeline
├── cleaning_pipeline.py      # Data cleaning workflow
├── local_inference.py        # Run model predictions
├── test_project.py          # Comprehensive test suite
├── nasa_apis.py             # NASA API configuration
├── requirements.txt         # Python dependencies
├── setup.sh                 # Setup automation script
│
├── data/                    # Data storage
│   ├── climate_model_ready_transformed.csv  # ML-ready data (1,800 samples)
│   ├── model_configuration.json             # Model config
│   ├── locations_major_cities.csv           # Location data
│   └── raw/                                 # Raw API responses
│
├── models/                  # Trained models
│   ├── climate_lstm_model.keras             # LSTM model (3.79 MB)
│   ├── lstm_scaler.pkl                      # Feature scaler
│   ├── lstm_model_metadata.json             # Performance metrics
│   └── lstm_training_history.pkl            # Training logs
│
├── src/                     # Source modules
│   ├── location_grid.py     # Location grid generator
│   ├── data_fetcher.py      # NASA API data fetcher
│   ├── data_processor.py    # Data processing
│   └── data_cleaner.py      # Data cleaning
│
├── results/                 # Model outputs
│   ├── climate_predictions_lstm_local.csv
│   └── local_predictions_visualization.png
│
└── venv/                    # Virtual environment
```

## 🌍 Location Grids

The pipeline supports three types of location grids:

### 1. Major Cities (--grid cities)
- 10 major cities worldwide
- Best for: Quick testing, city-specific predictions
- Locations: New Delhi, Mumbai, Bangalore, Chennai, Kolkata, NYC, LA, London, Tokyo, Sydney

### 2. India Regional Grid (--grid india)
- 2° x 2° grid covering India
- Bounds: 8°N-37°N, 68°E-97°E
- ~300 locations
- Best for: Regional India climate modeling

### 3. Global Grid (--grid global)
- 10° x 10° global coverage
- ~650 locations
- Best for: Global climate trend analysis

## 📊 Pipeline Steps

### Step 1: Generate Location Grid
Automatically generates latitude/longitude coordinates based on selected grid type.

### Step 2: Fetch Climate Data
For each location, fetches NASA POWER data:
- **Temporal Monthly API** (2005-2024)
  - 7 parameters: T2M, T2M_MAX, T2M_MIN, PRECTOTCORR, ALLSKY_SFC_SW_DWN, RH2M, QV2M
  - Monthly time-series data

### Step 3: Process & Create Master Dataset
- Merges location metadata
- Calculates derived features:
  - T2M_range (temperature range)
  - heat_index (simplified)
  - precip_log (log-transformed precipitation)
  - Season and month indicators
- Optional: Temperature anomalies (requires baseline period data)
- Optional: Lag features for time-series modeling

## 🔧 Usage Examples

### Make Predictions
```bash
python local_inference.py
# Generates predictions for all 1,800 samples
# Outputs: results/climate_predictions_lstm_local.csv
#          results/local_predictions_visualization.png
```

### Test Everything
```bash
python test_project.py
# Runs 8 comprehensive tests
# Verifies all components work correctly
```

### Fetch New Data - Test Run (3 locations, 2 years)
```bash
python main_pipeline.py --grid cities --start 2020 --end 2021 --test
```

### Fetch New Data - Major Cities (Full Period)
```bash
python main_pipeline.py --grid cities --start 2010 --end 2024
```

### India Regional Grid (Limited Locations)
```bash
python main_pipeline.py --grid india --start 2010 --end 2024 --max-locations 50
```

## 📈 Output Datasets

### 1. Location Grid CSV
Format: `locations_[grid_type].csv`
- location_id, latitude, longitude, grid_type, description

### 2. Raw Climate Data CSV
Format: `climate_data_[grid_type]_[start]_[end].csv`
- All raw NASA POWER parameters
- time, lat, lon, location_id, climate parameters

### 3. Master Dataset CSV
Format: `climate_master_[grid_type]_[start]_[end].csv`
- Processed data with derived features
- Ready for ML model training
- Columns: 19 features including:
  - Original climate parameters (7)
  - Location metadata (lat, lon, location_id, description)
  - Time features (time, month, season, year)
  - Derived features (T2M_range, heat_index, precip_log)

## 🌡️ Climate Parameters

### Core Parameters
1. **T2M** - Mean temperature at 2m (°C)
2. **T2M_MAX** - Maximum temperature at 2m (°C)
3. **T2M_MIN** - Minimum temperature at 2m (°C)

### Hydrology
4. **PRECTOTCORR** - Corrected precipitation (mm/month)

### Radiation
5. **ALLSKY_SFC_SW_DWN** - Surface shortwave radiation (W/m²)

### Moisture
6. **RH2M** - Relative humidity at 2m (%)
7. **QV2M** - Specific humidity at 2m (g/kg)

## ⚙️ Command-Line Options

```
Options:
  --grid {global,india,cities}
                        Type of location grid (default: cities)
  --start YEAR          Start year for data collection (default: 2010)
  --end YEAR            End year for data collection (default: 2024)
  --test                Test mode: only fetch 3 locations
  --max-locations N     Maximum number of locations to fetch
```

## 📝 Notes

- API requests are rate-limited (1 second delay between requests)
- NASA POWER API includes month "13" as annual average (automatically filtered)
- Raw JSON responses saved in `data/raw/` for debugging
- Test mode recommended before full runs
- Global grid (~650 locations) takes 10-12 hours to complete

## 🔄 Next Steps

After generating the master dataset:
1. Perform exploratory data analysis (EDA)
2. Engineer additional features
3. Train ML models (regression, time-series, etc.)
4. Validate predictions
5. Deploy models

## 📦 Dependencies

All dependencies are installed in `venv/`:
- pandas, numpy - Data processing
- tensorflow, keras - ML model
- scikit-learn - ML utilities  
- matplotlib, seaborn - Visualization
- joblib - Model persistence
- requests - API calls

## 🔄 Workflow

1. **Data Collection** → `main_pipeline.py`
2. **Data Cleaning** → `cleaning_pipeline.py`
3. **Model Training** → (done in Google Colab)
4. **Predictions** → `local_inference.py`
5. **Testing** → `test_project.py`

````
