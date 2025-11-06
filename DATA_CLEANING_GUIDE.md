# Data Cleaning Pipeline - NASA Climate Data

This document explains the comprehensive data cleaning process implemented for the NASA Climate Data project.

## Overview

The data cleaning pipeline follows best practices for preparing climate data for machine learning:

1. **Handle Missing Values** - Intelligent strategies for filling/removing missing data
2. **Handle Infinite Values** - Replace or remove inf/-inf values
3. **Remove Duplicates** - Eliminate duplicate rows
4. **Standardize Format** - Consistent data types and formatting
5. **Detect and Treat Outliers** - Identify and handle extreme values
6. **Scale and Normalize** - Standardize feature ranges
7. **Encode Categorical** - Convert categorical variables to numeric
8. **Validate and Reconcile** - Final quality checks

## Pipeline Stages

### Stage 1: After Fetching - Data Quality Analysis

**Purpose**: Understand data quality issues before cleaning

```python
from src.data_cleaner import DataCleaner

cleaner = DataCleaner()
quality_report = cleaner.analyze_data_quality(df)
```

**Metrics Analyzed**:
- Missing value counts and percentages
- Infinite value detection
- Duplicate row identification
- Data type validation
- Memory usage assessment

### Stage 2: Cleaning Operations

#### 2.1 Handle Missing Values

**Strategies Available**:
- `smart` (default): Intelligent context-aware filling
  - Time-series columns: Linear interpolation
  - Categorical columns: Forward fill + mode
  - Remaining: Median imputation
- `mean`: Fill with column mean
- `median`: Fill with column median
- `mode`: Fill with most frequent value
- `forward_fill`: Propagate last valid observation
- `interpolate`: Linear interpolation
- `drop`: Remove rows with missing values

```python
df_cleaned = cleaner.handle_missing_values(df, strategy='smart')
```

#### 2.2 Handle Infinite Values

**Strategies**:
- `replace` (default): Replace inf with NaN, then fill with median
- `drop`: Remove rows with infinite values
- `clip`: Clip to min/max finite values

```python
df_cleaned = cleaner.handle_inf_values(df, strategy='replace')
```

#### 2.3 Remove Duplicates

```python
df_cleaned = cleaner.remove_duplicates(df, subset=None, keep='first')
```

Options:
- `subset`: Columns to check (None = all columns)
- `keep`: 'first', 'last', or False (remove all duplicates)

#### 2.4 Standardize Format

- Ensures datetime columns are properly formatted
- Standardizes string columns (lowercase, strip whitespace)
- Rounds numeric values to reasonable precision

```python
df_cleaned = cleaner.standardize_format(df)
```

### Stage 3: Outlier Detection and Treatment

#### 3.1 Detect Outliers

**Methods**:
- `iqr` (Interquartile Range): Standard method, threshold=1.5 or 3.0
- `zscore`: Z-score method, threshold=3.0 (3 standard deviations)

```python
outlier_info = cleaner.detect_outliers(df, method='iqr', threshold=3.0)
```

#### 3.2 Treat Outliers

**Treatment Methods**:
- `clip` (recommended): Cap values at boundaries
- `remove`: Remove outlier rows
- `winsorize`: Cap at percentiles (5th and 95th)
- `log_transform`: Apply log transformation for skewed data

```python
df_cleaned = cleaner.treat_outliers(df, method='clip', outlier_info=outlier_info)
```

### Stage 4: Feature Scaling and Encoding

#### 4.1 Scale Features

**Scaling Methods**:
- `standard` (StandardScaler): Mean=0, Std=1
  - Best for: Normal distributions, linear models
  - Formula: (x - mean) / std
  
- `minmax` (MinMaxScaler): Scale to [0, 1] range
  - Best for: Neural networks, bounded data
  - Formula: (x - min) / (max - min)
  
- `robust` (RobustScaler): Uses median and IQR
  - Best for: Data with outliers
  - Formula: (x - median) / IQR

```python
df_scaled = cleaner.scale_features(df, method='standard', fit=True)
```

**Important**: 
- Set `fit=True` for training data (fits the scaler)
- Set `fit=False` for test/inference data (uses fitted scaler)

#### 4.2 Encode Categorical Variables

**Encoding Methods**:
- `onehot`: One-hot encoding (creates binary columns)
  - Best for: Nominal categories, few unique values
  
- `label`: Label encoding (converts to integers)
  - Best for: Ordinal categories, many unique values

```python
df_encoded = cleaner.encode_categorical(df, columns=['season', 'grid_type'], 
                                       method='label', fit=True)
```

### Stage 5: Validation and Artifact Saving

#### 5.1 Validate Cleaned Data

```python
validation_report = cleaner.validate_data(df_cleaned)
```

Checks:
- No missing values
- No infinite values
- No duplicates
- Correct data types
- Statistical summary

#### 5.2 Save Artifacts for Inference

```python
cleaner.save_artifacts(prefix='cleaning')
```

Saves:
- `models/cleaning_scalers.pkl` - Fitted scalers
- `models/cleaning_encoders.pkl` - Fitted encoders
- `models/cleaning_report.json` - Cleaning metadata

**For Inference**:
```python
cleaner.load_artifacts(prefix='cleaning')
df_new = cleaner.scale_features(df_new, method='standard', fit=False)
```

## Complete Pipeline Usage

### For Training Data

```python
from src.data_cleaner import DataCleaner
import pandas as pd

# Initialize
cleaner = DataCleaner(data_dir='data', models_dir='models')

# Load data
df = pd.read_csv('data/climate_master_cities_2010_2024.csv')

# Run complete pipeline
cleaned_df = cleaner.clean_pipeline(
    df=df,
    for_training=True,
    scale_method='standard',
    handle_outliers=True,
    outlier_method='clip'
)

# Save cleaned data
cleaned_df.to_csv('data/climate_cleaned_cities_2010_2024.csv', index=False)
```

### For Inference/Test Data

```python
# Load artifacts
cleaner = DataCleaner(data_dir='data', models_dir='models')
cleaner.load_artifacts(prefix='cleaning')

# Load new data
df_new = pd.read_csv('new_data.csv')

# Apply same transformations
cleaned_new = cleaner.clean_pipeline(
    df=df_new,
    for_training=False,  # Use existing scalers/encoders
    scale_method='standard',
    handle_outliers=False  # Don't fit new outlier bounds
)
```

## Command-Line Usage

### Basic Cleaning

```bash
python cleaning_pipeline.py
```

### Custom Parameters

```bash
# Use robust scaling
python cleaning_pipeline.py --scale-method robust

# Remove outliers instead of clipping
python cleaning_pipeline.py --outlier-method remove

# MinMax scaling with winsorization
python cleaning_pipeline.py --scale-method minmax --outlier-method winsorize

# Skip outlier handling
python cleaning_pipeline.py --no-handle-outliers

# Custom input/output files
python cleaning_pipeline.py --input data/my_data.csv --output data/my_cleaned.csv
```

## Pipeline Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATA                                  │
│              (climate_master_*.csv)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: DATA QUALITY ANALYSIS                              │
│  - Analyze missing values                                    │
│  - Detect infinite values                                    │
│  - Find duplicates                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: HANDLE MISSING VALUES                              │
│  - Smart interpolation for time-series                       │
│  - Forward fill for categorical                              │
│  - Median imputation for remaining                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: HANDLE INFINITE VALUES                             │
│  - Replace inf with NaN                                      │
│  - Fill with median                                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: REMOVE DUPLICATES                                  │
│  - Remove duplicate rows                                     │
│  - Keep first occurrence                                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: STANDARDIZE FORMAT                                 │
│  - Format datetime columns                                   │
│  - Standardize strings                                       │
│  - Round numeric precision                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 6: DETECT & TREAT OUTLIERS                            │
│  - IQR method (threshold=3.0)                               │
│  - Clip to bounds                                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 7: ENCODE CATEGORICAL                                 │
│  - Label encoding for season, grid_type                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 8: SCALE FEATURES                                     │
│  - StandardScaler (mean=0, std=1)                           │
│  - Fit and save scaler                                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 9: VALIDATE                                           │
│  - Check for remaining issues                                │
│  - Generate quality report                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 10: SAVE ARTIFACTS                                    │
│  - Cleaned data (climate_cleaned_*.csv)                      │
│  - Scalers (models/cleaning_scalers.pkl)                     │
│  - Encoders (models/cleaning_encoders.pkl)                   │
│  - Report (models/cleaning_report.json)                      │
└─────────────────────────────────────────────────────────────┘
```

## Best Practices

### 1. Order Matters

Follow this sequence:
1. Handle missing/inf values first
2. Remove duplicates
3. Detect outliers (after data is complete)
4. Treat outliers
5. Encode categorical
6. Scale features (always last)

### 2. Save Artifacts for Production

Always save fitted scalers and encoders:
```python
cleaner.save_artifacts(prefix='cleaning')
```

Then load them for inference:
```python
cleaner.load_artifacts(prefix='cleaning')
```

### 3. Scaling Method Selection

| Data Type | Recommended Method |
|-----------|-------------------|
| Normal distribution | Standard |
| Neural networks | MinMax |
| Data with outliers | Robust |
| Tree-based models | None (not required) |

### 4. Outlier Treatment Selection

| Scenario | Recommended Method |
|----------|-------------------|
| Keep all data | Clip |
| Remove extreme values | Remove |
| Robust statistics | Winsorize |
| Skewed distribution | Log Transform |

### 5. Test/Train Consistency

**Critical**: Use the same transformations on test data!

```python
# Training
cleaner.clean_pipeline(train_df, for_training=True)
cleaner.save_artifacts()

# Testing
cleaner.load_artifacts()
cleaner.clean_pipeline(test_df, for_training=False)
```

## Monitoring and Validation

### Cleaning Report

Generated automatically at `models/cleaning_report.json`:

```json
{
  "missing_values": {
    "initial": 150,
    "final": 0,
    "strategy": "smart"
  },
  "inf_values": {
    "initial": 5,
    "final": 0,
    "strategy": "replace"
  },
  "duplicates": {
    "initial_count": 23,
    "removed": 23
  },
  "outliers": {
    "method": "iqr",
    "threshold": 3.0,
    "columns": {...}
  },
  "scaling": {
    "method": "standard",
    "columns": [...]
  }
}
```

### Quality Metrics

Check these metrics after cleaning:
- ✓ Missing values = 0
- ✓ Infinite values = 0
- ✓ Duplicates = 0
- ✓ Outliers handled
- ✓ All features scaled
- ✓ Categorical encoded

## Troubleshooting

### Issue: "Scaler not found"
**Solution**: Ensure you saved artifacts after training:
```python
cleaner.save_artifacts(prefix='cleaning')
```

### Issue: "Shape mismatch after encoding"
**Solution**: Use `fit=False` for test data to maintain consistency:
```python
cleaner.encode_categorical(test_df, fit=False)
```

### Issue: "Too many outliers removed"
**Solution**: Use 'clip' instead of 'remove':
```python
cleaner.treat_outliers(df, method='clip')
```

### Issue: "Memory error with large dataset"
**Solution**: Process in chunks or use robust scaling (more memory efficient):
```python
cleaner.scale_features(df, method='robust')
```

## Files Generated

After running the cleaning pipeline:

```
project/
├── data/
│   ├── climate_master_cities_2010_2024.csv          # Original
│   ├── climate_cleaned_cities_2010_2024.csv         # Cleaned
│   └── climate_cleaned_cities_2010_2024_summary.txt # Summary
├── models/
│   ├── cleaning_scalers.pkl    # Fitted scalers (for inference)
│   ├── cleaning_encoders.pkl   # Fitted encoders (for inference)
│   └── cleaning_report.json    # Detailed cleaning report
└── src/
    └── data_cleaner.py         # Cleaning module
```

## Next Steps

After cleaning:

1. **Feature Engineering** - Create domain-specific features
2. **Train/Test Split** - Divide data for model training
3. **Model Training** - Train ML models on cleaned data
4. **Model Evaluation** - Validate model performance
5. **Deployment** - Use saved artifacts in production

## References

- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Pandas Data Cleaning](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Feature Scaling Best Practices](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html)
