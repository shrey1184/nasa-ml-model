"""
Data Cleaning Pipeline
Handles missing values, outliers, duplicates, standardization, scaling, and validation
Implements comprehensive data quality checks and transformations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """
    Comprehensive data cleaning pipeline for climate data
    Handles: missing values, inf values, duplicates, outliers, scaling, encoding, validation
    """
    
    def __init__(self, data_dir='data', models_dir='models'):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Store cleaning metadata
        self.cleaning_report = {
            'missing_values': {},
            'inf_values': {},
            'duplicates': {},
            'outliers': {},
            'scaling': {},
            'encoding': {},
            'validation': {}
        }
        
        # Store fitted transformers
        self.scalers = {}
        self.encoders = {}
        
    def analyze_data_quality(self, df):
        """
        Comprehensive data quality analysis
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        print("\n" + "="*70)
        print("DATA QUALITY ANALYSIS")
        print("="*70)
        
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        }
        
        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        report['missing_values'] = {
            col: {'count': int(missing[col]), 'percent': float(missing_pct[col])}
            for col in df.columns if missing[col] > 0
        }
        
        # Infinite values (for numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = int(inf_count)
        report['inf_values'] = inf_counts
        
        # Duplicates
        duplicate_rows = df.duplicated().sum()
        report['duplicates'] = {
            'count': int(duplicate_rows),
            'percent': float((duplicate_rows / len(df)) * 100)
        }
        
        # Data types
        report['data_types'] = df.dtypes.astype(str).to_dict()
        
        # Print summary
        print(f"\nTotal Rows: {report['total_rows']:,}")
        print(f"Total Columns: {report['total_columns']}")
        print(f"Memory Usage: {report['memory_usage_mb']:.2f} MB")
        
        if report['missing_values']:
            print(f"\n⚠ Missing Values Found in {len(report['missing_values'])} columns:")
            for col, stats in sorted(report['missing_values'].items(), 
                                    key=lambda x: x[1]['percent'], reverse=True)[:10]:
                print(f"  {col}: {stats['count']} ({stats['percent']:.2f}%)")
        else:
            print("\n✓ No missing values found")
        
        if report['inf_values']:
            print(f"\n⚠ Infinite Values Found in {len(report['inf_values'])} columns:")
            for col, count in report['inf_values'].items():
                print(f"  {col}: {count}")
        else:
            print("\n✓ No infinite values found")
        
        if report['duplicates']['count'] > 0:
            print(f"\n⚠ Duplicates: {report['duplicates']['count']} rows "
                  f"({report['duplicates']['percent']:.2f}%)")
        else:
            print("\n✓ No duplicate rows found")
        
        print("="*70)
        
        return report
    
    def handle_missing_values(self, df, strategy='smart', fill_value=None):
        """
        Handle missing values with various strategies
        
        Args:
            df: Input DataFrame
            strategy: 'smart', 'mean', 'median', 'mode', 'forward_fill', 'drop', 'interpolate'
            fill_value: Custom fill value (for strategy='custom')
            
        Returns:
            Cleaned DataFrame
        """
        print("\n" + "="*70)
        print("HANDLING MISSING VALUES")
        print("="*70)
        print(f"Strategy: {strategy}")
        
        df = df.copy()
        initial_missing = df.isnull().sum().sum()
        
        if initial_missing == 0:
            print("✓ No missing values to handle")
            return df
        
        print(f"\nInitial missing values: {initial_missing}")
        
        if strategy == 'smart':
            # Smart strategy: different approaches for different column types
            
            # For time-series numeric data: interpolate
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            time_series_cols = [col for col in numeric_cols 
                              if col not in ['location_id', 'year', 'month', 'latitude', 'longitude']]
            
            if time_series_cols:
                print(f"\n  Interpolating {len(time_series_cols)} time-series columns...")
                # Sort by location and time for proper interpolation
                if 'location_id' in df.columns and 'time' in df.columns:
                    df = df.sort_values(['location_id', 'time'])
                    for col in time_series_cols:
                        df[col] = df.groupby('location_id')[col].transform(
                            lambda x: x.interpolate(method='linear', limit_direction='both')
                        )
                else:
                    for col in time_series_cols:
                        df[col] = df[col].interpolate(method='linear', limit_direction='both')
            
            # For categorical: forward fill then mode
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                print(f"  Forward filling {len(categorical_cols)} categorical columns...")
                for col in categorical_cols:
                    df[col] = df[col].fillna(method='ffill').fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            
            # For remaining missing: use median
            remaining_missing = df.isnull().sum().sum()
            if remaining_missing > 0:
                print(f"  Filling remaining {remaining_missing} missing values with median...")
                df = df.fillna(df.median(numeric_only=True))
        
        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        elif strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        elif strategy == 'mode':
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else fill_value
                    df[col] = df[col].fillna(mode_val)
        
        elif strategy == 'forward_fill':
            df = df.fillna(method='ffill')
        
        elif strategy == 'interpolate':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
        
        elif strategy == 'drop':
            df = df.dropna()
        
        final_missing = df.isnull().sum().sum()
        print(f"\nFinal missing values: {final_missing}")
        print(f"✓ Removed/filled {initial_missing - final_missing} missing values")
        
        self.cleaning_report['missing_values'] = {
            'initial': int(initial_missing),
            'final': int(final_missing),
            'strategy': strategy
        }
        
        print("="*70)
        return df
    
    def handle_inf_values(self, df, strategy='replace'):
        """
        Handle infinite values
        
        Args:
            df: Input DataFrame
            strategy: 'replace' (with NaN then fill), 'drop', or 'clip'
            
        Returns:
            Cleaned DataFrame
        """
        print("\n" + "="*70)
        print("HANDLING INFINITE VALUES")
        print("="*70)
        
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        inf_count = sum(np.isinf(df[col]).sum() for col in numeric_cols)
        
        if inf_count == 0:
            print("✓ No infinite values found")
            return df
        
        print(f"Found {inf_count} infinite values")
        print(f"Strategy: {strategy}")
        
        if strategy == 'replace':
            # Replace inf with NaN, then use median
            for col in numeric_cols:
                inf_mask = np.isinf(df[col])
                if inf_mask.any():
                    print(f"  Replacing {inf_mask.sum()} inf values in {col}")
                    df.loc[inf_mask, col] = np.nan
                    df[col] = df[col].fillna(df[col].median())
        
        elif strategy == 'drop':
            initial_len = len(df)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            print(f"  Dropped {initial_len - len(df)} rows")
        
        elif strategy == 'clip':
            # Clip to min/max non-inf values
            for col in numeric_cols:
                if np.isinf(df[col]).any():
                    finite_vals = df[col][np.isfinite(df[col])]
                    if len(finite_vals) > 0:
                        min_val, max_val = finite_vals.min(), finite_vals.max()
                        df[col] = df[col].clip(min_val, max_val)
        
        final_inf = sum(np.isinf(df[col]).sum() for col in numeric_cols)
        print(f"✓ Remaining infinite values: {final_inf}")
        
        self.cleaning_report['inf_values'] = {
            'initial': int(inf_count),
            'final': int(final_inf),
            'strategy': strategy
        }
        
        print("="*70)
        return df
    
    def remove_duplicates(self, df, subset=None, keep='first'):
        """
        Remove duplicate rows
        
        Args:
            df: Input DataFrame
            subset: List of columns to check for duplicates (None = all columns)
            keep: 'first', 'last', or False (remove all duplicates)
            
        Returns:
            Cleaned DataFrame
        """
        print("\n" + "="*70)
        print("REMOVING DUPLICATES")
        print("="*70)
        
        initial_len = len(df)
        duplicates = df.duplicated(subset=subset, keep=keep).sum()
        
        if duplicates == 0:
            print("✓ No duplicate rows found")
            return df
        
        print(f"Found {duplicates} duplicate rows ({(duplicates/initial_len)*100:.2f}%)")
        
        if subset:
            print(f"Checking columns: {subset}")
        
        df = df.drop_duplicates(subset=subset, keep=keep)
        
        final_len = len(df)
        removed = initial_len - final_len
        
        print(f"✓ Removed {removed} duplicate rows")
        print(f"  Remaining rows: {final_len}")
        
        self.cleaning_report['duplicates'] = {
            'initial_count': int(duplicates),
            'removed': int(removed),
            'subset': subset,
            'keep': keep
        }
        
        print("="*70)
        return df
    
    def standardize_format(self, df):
        """
        Standardize data formats (dates, strings, etc.)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Standardized DataFrame
        """
        print("\n" + "="*70)
        print("STANDARDIZING FORMAT")
        print("="*70)
        
        df = df.copy()
        
        # Ensure datetime columns are properly formatted
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            print("✓ Standardized 'time' column to datetime")
        
        # Standardize string columns (lowercase, strip whitespace)
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            if col not in ['time']:  # Skip datetime-like columns
                df[col] = df[col].astype(str).str.strip().str.lower()
                print(f"✓ Standardized string column: {col}")
        
        # Round numeric columns to reasonable precision
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['location_id', 'year', 'month']:
                # Round to 4 decimal places for climate data
                df[col] = df[col].round(4)
        
        print("✓ Rounded numeric values to 4 decimal places")
        print("="*70)
        return df
    
    def detect_outliers(self, df, method='iqr', threshold=3.0):
        """
        Detect outliers in numeric columns
        
        Args:
            df: Input DataFrame
            method: 'iqr' (Interquartile Range) or 'zscore' (Z-score)
            threshold: For IQR: multiplier (1.5 standard), For Z-score: std deviations
            
        Returns:
            DataFrame with outlier flags
        """
        print("\n" + "="*70)
        print("DETECTING OUTLIERS")
        print("="*70)
        print(f"Method: {method.upper()}, Threshold: {threshold}")
        
        df = df.copy()
        
        # Columns to check for outliers (exclude IDs, timestamps, categorical)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['location_id', 'year', 'month', 'latitude', 'longitude']
        check_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        outlier_info = {}
        
        for col in check_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    outlier_info[col] = {
                        'count': int(outlier_count),
                        'percent': float((outlier_count / len(df)) * 100),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = z_scores > threshold
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    outlier_info[col] = {
                        'count': int(outlier_count),
                        'percent': float((outlier_count / len(df)) * 100),
                        'threshold': float(threshold)
                    }
        
        if outlier_info:
            print(f"\n⚠ Outliers detected in {len(outlier_info)} columns:")
            for col, info in sorted(outlier_info.items(), 
                                   key=lambda x: x[1]['count'], reverse=True)[:10]:
                print(f"  {col}: {info['count']} ({info['percent']:.2f}%)")
        else:
            print("\n✓ No significant outliers detected")
        
        self.cleaning_report['outliers'] = {
            'method': method,
            'threshold': threshold,
            'columns': outlier_info
        }
        
        print("="*70)
        return outlier_info
    
    def treat_outliers(self, df, method='clip', outlier_info=None, 
                      detection_method='iqr', threshold=3.0):
        """
        Treat outliers in the data
        
        Args:
            df: Input DataFrame
            method: 'clip', 'remove', 'winsorize', or 'log_transform'
            outlier_info: Pre-computed outlier information (if None, will detect)
            detection_method: Method to detect outliers ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Treated DataFrame
        """
        print("\n" + "="*70)
        print("TREATING OUTLIERS")
        print("="*70)
        print(f"Treatment method: {method}")
        
        df = df.copy()
        
        # Detect outliers if not provided
        if outlier_info is None:
            outlier_info = self.detect_outliers(df, method=detection_method, threshold=threshold)
        
        if not outlier_info:
            print("✓ No outliers to treat")
            return df
        
        initial_len = len(df)
        
        if method == 'clip':
            # Clip values to bounds
            for col, info in outlier_info.items():
                if 'lower_bound' in info and 'upper_bound' in info:
                    df[col] = df[col].clip(info['lower_bound'], info['upper_bound'])
                    print(f"  Clipped {col} to [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
        
        elif method == 'remove':
            # Remove outlier rows
            mask = pd.Series([True] * len(df), index=df.index)
            for col, info in outlier_info.items():
                if 'lower_bound' in info and 'upper_bound' in info:
                    col_mask = (df[col] >= info['lower_bound']) & (df[col] <= info['upper_bound'])
                    mask = mask & col_mask
            
            df = df[mask]
            removed = initial_len - len(df)
            print(f"  Removed {removed} outlier rows ({(removed/initial_len)*100:.2f}%)")
        
        elif method == 'winsorize':
            # Winsorize (cap at percentiles)
            from scipy.stats.mstats import winsorize
            for col in outlier_info.keys():
                df[col] = winsorize(df[col], limits=[0.05, 0.05])  # 5th and 95th percentiles
                print(f"  Winsorized {col}")
        
        elif method == 'log_transform':
            # Log transformation for skewed data
            for col in outlier_info.keys():
                if (df[col] > 0).all():  # Only for positive values
                    df[f'{col}_log'] = np.log1p(df[col])
                    print(f"  Created log-transformed column: {col}_log")
        
        print(f"✓ Outlier treatment complete")
        print("="*70)
        return df
    
    def scale_features(self, df, method='standard', columns=None, fit=True):
        """
        Scale/normalize features
        
        Args:
            df: Input DataFrame
            method: 'standard' (StandardScaler), 'minmax' (MinMaxScaler), 
                   'robust' (RobustScaler)
            columns: List of columns to scale (None = all numeric except IDs)
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Scaled DataFrame
        """
        print("\n" + "="*70)
        print("SCALING FEATURES")
        print("="*70)
        print(f"Method: {method}")
        print(f"Mode: {'Fit & Transform' if fit else 'Transform Only'}")
        
        df = df.copy()
        
        # Determine columns to scale
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            exclude_cols = ['location_id', 'year', 'month']
            columns = [col for col in numeric_cols if col not in exclude_cols]
        
        print(f"Scaling {len(columns)} columns")
        
        # Select scaler
        if method == 'standard':
            scaler_class = StandardScaler
        elif method == 'minmax':
            scaler_class = MinMaxScaler
        elif method == 'robust':
            scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit or load scaler
        scaler_name = f'{method}_scaler'
        
        if fit:
            # Fit new scaler
            scaler = scaler_class()
            df[columns] = scaler.fit_transform(df[columns])
            self.scalers[scaler_name] = scaler
            print(f"✓ Fitted and applied {method} scaler")
        else:
            # Use existing scaler
            if scaler_name not in self.scalers:
                raise ValueError(f"Scaler '{scaler_name}' not found. Must fit first.")
            scaler = self.scalers[scaler_name]
            df[columns] = scaler.transform(df[columns])
            print(f"✓ Applied existing {method} scaler")
        
        self.cleaning_report['scaling'] = {
            'method': method,
            'columns': columns,
            'scaler_name': scaler_name
        }
        
        print("="*70)
        return df
    
    def encode_categorical(self, df, columns=None, method='onehot', fit=True):
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            columns: List of categorical columns to encode (None = auto-detect)
            method: 'onehot', 'label', or 'target'
            fit: Whether to fit the encoder (True for training, False for inference)
            
        Returns:
            Encoded DataFrame
        """
        print("\n" + "="*70)
        print("ENCODING CATEGORICAL FEATURES")
        print("="*70)
        print(f"Method: {method}")
        
        df = df.copy()
        
        # Auto-detect categorical columns
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            # Exclude certain columns
            exclude = ['time', 'description']
            columns = [col for col in columns if col not in exclude]
        
        if not columns:
            print("✓ No categorical columns to encode")
            return df
        
        print(f"Encoding {len(columns)} columns: {columns}")
        
        if method == 'onehot':
            for col in columns:
                encoder_name = f'{col}_onehot_encoder'
                
                if fit:
                    # Create dummy variables
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
                    
                    # Store column names for inference
                    self.encoders[encoder_name] = dummies.columns.tolist()
                    print(f"  ✓ One-hot encoded {col} -> {len(dummies.columns)} columns")
                else:
                    # Use stored column names
                    if encoder_name not in self.encoders:
                        raise ValueError(f"Encoder '{encoder_name}' not found. Must fit first.")
                    
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    # Ensure same columns as training
                    for dummy_col in self.encoders[encoder_name]:
                        if dummy_col not in dummies.columns:
                            dummies[dummy_col] = 0
                    dummies = dummies[self.encoders[encoder_name]]
                    
                    df = pd.concat([df, dummies], axis=1)
                    df = df.drop(columns=[col])
        
        elif method == 'label':
            for col in columns:
                encoder_name = f'{col}_label_encoder'
                
                if fit:
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
                    self.encoders[encoder_name] = encoder
                    print(f"  ✓ Label encoded {col}")
                else:
                    if encoder_name not in self.encoders:
                        raise ValueError(f"Encoder '{encoder_name}' not found. Must fit first.")
                    encoder = self.encoders[encoder_name]
                    df[col] = encoder.transform(df[col].astype(str))
        
        self.cleaning_report['encoding'] = {
            'method': method,
            'columns': columns
        }
        
        print("="*70)
        return df
    
    def validate_data(self, df):
        """
        Validate cleaned data
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Validation report
        """
        print("\n" + "="*70)
        print("VALIDATING CLEANED DATA")
        print("="*70)
        
        validation = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        }
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = sum(np.isinf(df[col]).sum() for col in numeric_cols)
        validation['infinite_values'] = int(inf_count)
        
        # Basic statistics
        validation['statistics'] = df.describe().to_dict()
        
        # Print validation summary
        print(f"\n✓ Total Rows: {validation['total_rows']:,}")
        print(f"✓ Total Columns: {validation['total_columns']}")
        print(f"✓ Missing Values: {validation['missing_values']}")
        print(f"✓ Duplicate Rows: {validation['duplicate_rows']}")
        print(f"✓ Infinite Values: {validation['infinite_values']}")
        print(f"✓ Numeric Columns: {validation['numeric_columns']}")
        print(f"✓ Categorical Columns: {validation['categorical_columns']}")
        
        # Validation checks
        all_checks_passed = True
        
        if validation['missing_values'] > 0:
            print("\n⚠ WARNING: Missing values still present")
            all_checks_passed = False
        
        if validation['infinite_values'] > 0:
            print("\n⚠ WARNING: Infinite values still present")
            all_checks_passed = False
        
        if all_checks_passed:
            print("\n✓ ALL VALIDATION CHECKS PASSED")
        
        self.cleaning_report['validation'] = validation
        
        print("="*70)
        return validation
    
    def save_artifacts(self, prefix='cleaning'):
        """
        Save scalers, encoders, and cleaning report
        
        Args:
            prefix: Prefix for saved files
        """
        print("\n" + "="*70)
        print("SAVING ARTIFACTS")
        print("="*70)
        
        # Save scalers
        if self.scalers:
            scaler_path = self.models_dir / f'{prefix}_scalers.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers, f)
            print(f"✓ Saved scalers to {scaler_path}")
        
        # Save encoders
        if self.encoders:
            encoder_path = self.models_dir / f'{prefix}_encoders.pkl'
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.encoders, f)
            print(f"✓ Saved encoders to {encoder_path}")
        
        # Save cleaning report
        report_path = self.models_dir / f'{prefix}_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.cleaning_report, f, indent=2, default=str)
        print(f"✓ Saved cleaning report to {report_path}")
        
        print("="*70)
    
    def load_artifacts(self, prefix='cleaning'):
        """
        Load saved scalers and encoders for inference
        
        Args:
            prefix: Prefix for saved files
        """
        print("\n" + "="*70)
        print("LOADING ARTIFACTS")
        print("="*70)
        
        # Load scalers
        scaler_path = self.models_dir / f'{prefix}_scalers.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scalers = pickle.load(f)
            print(f"✓ Loaded scalers from {scaler_path}")
        
        # Load encoders
        encoder_path = self.models_dir / f'{prefix}_encoders.pkl'
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                self.encoders = pickle.load(f)
            print(f"✓ Loaded encoders from {encoder_path}")
        
        print("="*70)
    
    def clean_pipeline(self, df, for_training=True, scale_method='standard',
                      handle_outliers=True, outlier_method='clip'):
        """
        Complete cleaning pipeline
        
        Args:
            df: Input DataFrame
            for_training: True for training data, False for inference
            scale_method: Scaling method ('standard', 'minmax', 'robust')
            handle_outliers: Whether to detect and treat outliers
            outlier_method: How to treat outliers ('clip', 'remove', 'winsorize')
            
        Returns:
            Cleaned DataFrame
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE DATA CLEANING PIPELINE")
        print("="*80)
        print(f"Mode: {'TRAINING' if for_training else 'INFERENCE'}")
        print(f"Initial shape: {df.shape}")
        print("="*80)
        
        # Step 1: Analyze data quality
        self.analyze_data_quality(df)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df, strategy='smart')
        
        # Step 3: Handle infinite values
        df = self.handle_inf_values(df, strategy='replace')
        
        # Step 4: Remove duplicates
        df = self.remove_duplicates(df, subset=None, keep='first')
        
        # Step 5: Standardize format
        df = self.standardize_format(df)
        
        # Step 6: Detect and treat outliers (only for training)
        if handle_outliers and for_training:
            outlier_info = self.detect_outliers(df, method='iqr', threshold=3.0)
            if outlier_info:
                df = self.treat_outliers(df, method=outlier_method, outlier_info=outlier_info)
        
        # Step 7: Encode categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in ['time', 'description']]
        
        if categorical_cols:
            df = self.encode_categorical(df, columns=categorical_cols, 
                                        method='label', fit=for_training)
        
        # Step 8: Scale features
        df = self.scale_features(df, method=scale_method, fit=for_training)
        
        # Step 9: Validate
        validation = self.validate_data(df)
        
        # Step 10: Save artifacts (only for training)
        if for_training:
            self.save_artifacts(prefix='cleaning')
        
        print("\n" + "="*80)
        print("CLEANING PIPELINE COMPLETE")
        print("="*80)
        print(f"Final shape: {df.shape}")
        print(f"Rows removed: {self.cleaning_report.get('duplicates', {}).get('removed', 0)}")
        print("="*80 + "\n")
        
        return df


def main():
    """Test data cleaning pipeline"""
    
    # Initialize cleaner
    cleaner = DataCleaner(data_dir='data', models_dir='models')
    
    # Load data
    data_file = Path('data/climate_master_cities_2010_2024.csv')
    
    if not data_file.exists():
        print(f"✗ Data file not found: {data_file}")
        print("  Run main_pipeline.py first to create master dataset")
        return
    
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Run cleaning pipeline
    cleaned_df = cleaner.clean_pipeline(
        df=df,
        for_training=True,
        scale_method='standard',
        handle_outliers=True,
        outlier_method='clip'
    )
    
    # Save cleaned data
    output_file = cleaner.data_dir / 'climate_cleaned_cities_2010_2024.csv'
    cleaned_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved cleaned data to {output_file}")
    
    # Display sample
    print("\nSample cleaned data (first 5 rows):")
    print(cleaned_df.head())
    
    print("\nCleaning Report Summary:")
    print(json.dumps(cleaner.cleaning_report, indent=2, default=str))


if __name__ == "__main__":
    main()
