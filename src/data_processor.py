"""
Data Processing and Merging Pipeline
Combines data from multiple APIs, calculates anomalies, and creates final dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from .data_cleaner import DataCleaner


class DataProcessor:
    """Process and merge climate data from multiple sources"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
    
    def calculate_anomalies(self, df, baseline_period=(1981, 2010)):
        """
        Calculate temperature anomalies relative to baseline period
        
        Args:
            df: DataFrame with time-series climate data
            baseline_period: Tuple of (start_year, end_year) for baseline
            
        Returns:
            DataFrame with anomaly columns added
        """
        df = df.copy()
        df['year'] = df['time'].dt.year
        
        # Filter baseline period
        baseline_mask = (df['year'] >= baseline_period[0]) & (df['year'] <= baseline_period[1])
        baseline_df = df[baseline_mask]
        
        # Calculate baseline means for each location and month
        df['month'] = df['time'].dt.month
        
        temp_params = ['T2M', 'T2M_MAX', 'T2M_MIN']
        
        for param in temp_params:
            if param in df.columns:
                # Calculate baseline climatology (mean for each location-month combination)
                baseline_clim = baseline_df.groupby(['location_id', 'month'])[param].mean()
                baseline_clim.name = f'{param}_baseline'
                
                # Merge baseline back to main df
                df = df.merge(
                    baseline_clim.reset_index(),
                    on=['location_id', 'month'],
                    how='left'
                )
                
                # Calculate anomaly
                df[f'{param}_anom'] = df[param] - df[f'{param}_baseline']
        
        # Drop temporary columns
        df = df.drop(columns=['year', 'month'] + [col for col in df.columns if col.endswith('_baseline')])
        
        return df
    
    def calculate_derived_features(self, df):
        """
        Calculate derived climate features
        
        Args:
            df: DataFrame with climate data
            
        Returns:
            DataFrame with additional derived features
        """
        df = df.copy()
        
        # Temperature range
        if 'T2M_MAX' in df.columns and 'T2M_MIN' in df.columns:
            df['T2M_range'] = df['T2M_MAX'] - df['T2M_MIN']
        
        # Heat index (simplified)
        if 'T2M' in df.columns and 'RH2M' in df.columns:
            # Simple heat index approximation
            df['heat_index'] = df['T2M'] + 0.1 * df['RH2M']
        
        # Precipitation intensity (if available)
        if 'PRECTOTCORR' in df.columns:
            df['precip_log'] = np.log1p(df['PRECTOTCORR'])  # log(1+x) transformation
        
        # Seasonal indicators
        df['month'] = df['time'].dt.month
        df['season'] = df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        
        # Year
        df['year'] = df['time'].dt.year
        
        return df
    
    def add_lag_features(self, df, lag_months=[1, 3, 6, 12]):
        """
        Add lagged features for time-series modeling
        
        Args:
            df: DataFrame with climate data
            lag_months: List of lag periods in months
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        df = df.sort_values(['location_id', 'time'])
        
        feature_cols = ['T2M', 'PRECTOTCORR', 'RH2M']
        
        for col in feature_cols:
            if col in df.columns:
                for lag in lag_months:
                    df[f'{col}_lag{lag}'] = df.groupby('location_id')[col].shift(lag)
        
        return df
    
    def merge_location_metadata(self, df, locations_df):
        """
        Merge location metadata into climate data
        
        Args:
            df: Climate data DataFrame
            locations_df: Location metadata DataFrame
            
        Returns:
            Merged DataFrame
        """
        return df.merge(
            locations_df[['location_id', 'grid_type', 'description']],
            on='location_id',
            how='left'
        )
    
    def create_master_dataset(self, climate_data_file, locations_file, 
                             add_anomalies=True, add_derived=True, add_lags=False):
        """
        Create master dataset with all features
        
        Args:
            climate_data_file: Path to raw climate data CSV
            locations_file: Path to locations metadata CSV
            add_anomalies: Whether to calculate temperature anomalies
            add_derived: Whether to add derived features
            add_lags: Whether to add lag features
            
        Returns:
            Processed master DataFrame
        """
        print("\n" + "="*70)
        print("CREATING MASTER DATASET")
        print("="*70)
        
        # Load data
        print(f"\n1. Loading climate data from {climate_data_file}...")
        df = pd.read_csv(climate_data_file)
        df['time'] = pd.to_datetime(df['time'])
        print(f"   ✓ Loaded {len(df)} rows")
        
        print(f"\n2. Loading location metadata from {locations_file}...")
        locations_df = pd.read_csv(locations_file)
        print(f"   ✓ Loaded {len(locations_df)} locations")
        
        # Merge location metadata
        print("\n3. Merging location metadata...")
        df = self.merge_location_metadata(df, locations_df)
        
        # Calculate anomalies
        if add_anomalies and 'T2M' in df.columns:
            print("\n4. Calculating temperature anomalies...")
            # Note: Need baseline period data for accurate anomalies
            print("   ⚠ Skipping anomalies (requires baseline period 1981-2010 data)")
            # df = self.calculate_anomalies(df)
        
        # Add derived features
        if add_derived:
            print("\n5. Adding derived features...")
            df = self.calculate_derived_features(df)
            print(f"   ✓ Added derived features")
        
        # Add lag features
        if add_lags:
            print("\n6. Adding lag features...")
            df = self.add_lag_features(df)
            print(f"   ✓ Added lag features")
            # Drop rows with NaN due to lagging
            initial_len = len(df)
            df = df.dropna()
            print(f"   ⚠ Dropped {initial_len - len(df)} rows with NaN values")
        
        print("\n" + "="*70)
        print("MASTER DATASET SUMMARY")
        print("="*70)
        print(f"Total rows: {len(df)}")
        print(f"Unique locations: {df['location_id'].nunique()}")
        print(f"Date range: {df['time'].min()} to {df['time'].max()}")
        print(f"Total features: {len(df.columns)}")
        print(f"\nFeature columns:")
        for col in df.columns:
            print(f"  - {col}")
        print("="*70 + "\n")
        
        return df
    
    def save_master_dataset(self, df, filename='climate_master_dataset.csv'):
        """Save master dataset to CSV"""
        output_path = self.data_dir / filename
        df.to_csv(output_path, index=False)
        print(f"✓ Saved master dataset to {output_path}")
        print(f"  Shape: {df.shape}")
        return output_path
    
    def clean_dataset(self, df, for_training=True, scale_method='standard',
                     handle_outliers=True, outlier_method='clip'):
        """
        Clean dataset using comprehensive cleaning pipeline
        
        Args:
            df: Input DataFrame
            for_training: True for training data, False for inference
            scale_method: Scaling method ('standard', 'minmax', 'robust')
            handle_outliers: Whether to detect and treat outliers
            outlier_method: How to treat outliers ('clip', 'remove', 'winsorize')
            
        Returns:
            Cleaned DataFrame
        """
        cleaner = DataCleaner(data_dir=self.data_dir, models_dir='models')
        
        cleaned_df = cleaner.clean_pipeline(
            df=df,
            for_training=for_training,
            scale_method=scale_method,
            handle_outliers=handle_outliers,
            outlier_method=outlier_method
        )
        
        return cleaned_df


def main():
    """Process and create master dataset"""
    
    processor = DataProcessor(data_dir='data')
    
    # Check if raw data exists
    climate_file = Path('data/climate_data_test.csv')
    locations_file = Path('data/locations_major_cities.csv')
    
    if not climate_file.exists():
        print(f"✗ Climate data file not found: {climate_file}")
        print("  Run data_fetcher.py first to fetch climate data")
        return
    
    if not locations_file.exists():
        print(f"✗ Locations file not found: {locations_file}")
        print("  Run location_grid.py first to generate location grids")
        return
    
    # Create master dataset
    master_df = processor.create_master_dataset(
        climate_data_file=climate_file,
        locations_file=locations_file,
        add_anomalies=False,  # Set to True if you have baseline data
        add_derived=True,
        add_lags=False  # Set to True for time-series models
    )
    
    # Save master dataset
    processor.save_master_dataset(master_df, 'climate_master_dataset.csv')
    
    # Display sample
    print("\nSample data (first 10 rows):")
    print(master_df.head(10))
    
    print("\nData types:")
    print(master_df.dtypes)
    
    print("\nStatistical summary:")
    print(master_df.describe())


if __name__ == "__main__":
    main()
