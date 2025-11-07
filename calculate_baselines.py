#!/usr/bin/env python3
"""
Calculate Climatological Baselines from Training Data

This script computes monthly climate normals (baselines) for temperature and 
precipitation from the actual training data (2010-2023) to be used for 
accurate anomaly calculations in future predictions.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'

def calculate_baselines():
    """
    Calculate climatological baselines from training data
    Returns baseline temperature and precipitation for each location-month combination
    """
    
    print("🔍 Calculating Climate Baselines from Training Data")
    print("=" * 60)
    
    # Load the master climate data (contains actual unadjusted values)
    data_file = DATA_DIR / 'climate_master_cities_2010_2024.csv'
    
    if not data_file.exists():
        print(f"❌ Error: {data_file} not found!")
        print("Looking for alternative data files...")
        
        # Try alternative file names
        alternatives = [
            'climate_cleaned_cities_2010_2024.csv',
            'climate_data_cities_2010_2024.csv'
        ]
        
        for alt_file in alternatives:
            alt_path = DATA_DIR / alt_file
            if alt_path.exists():
                print(f"✓ Found: {alt_file}")
                data_file = alt_path
                break
        else:
            print("❌ No suitable data file found!")
            return None
    
    print(f"📂 Reading data from: {data_file.name}")
    df = pd.read_csv(data_file)
    
    print(f"✓ Loaded {len(df)} records")
    print(f"✓ Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
    
    # Check for required columns
    required_cols = ['latitude', 'longitude', 'month', 'T2M', 'PRECTOTCORR']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return None
    
    print("\n📊 Calculating Monthly Climatological Normals...")
    print("-" * 60)
    
    # Round coordinates to 2 decimals to group nearby locations
    df['lat_rounded'] = df['latitude'].round(2)
    df['lon_rounded'] = df['longitude'].round(2)
    
    # Calculate baseline (mean) for each location-month combination
    baselines = df.groupby(['lat_rounded', 'lon_rounded', 'month']).agg({
        'T2M': 'mean',
        'PRECTOTCORR': 'mean',
        'latitude': 'first',  # Keep original precision
        'longitude': 'first'
    }).reset_index()
    
    # Rename for clarity
    baselines.rename(columns={
        'T2M': 'baseline_temp',
        'PRECTOTCORR': 'baseline_precip'
    }, inplace=True)
    
    print(f"✓ Calculated baselines for {len(baselines)} location-month combinations")
    print(f"✓ Unique locations: {len(baselines[['lat_rounded', 'lon_rounded']].drop_duplicates())}")
    
    # Show sample baselines
    print("\n📋 Sample Baselines:")
    print("-" * 60)
    sample = baselines.head(12)
    for _, row in sample.iterrows():
        month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][int(row['month']) - 1]
        print(f"Location ({row['latitude']:.2f}°, {row['longitude']:.2f}°) - {month_name}: "
              f"Temp={row['baseline_temp']:.2f}°C, Precip={row['baseline_precip']:.2f}mm")
    
    # Save to CSV
    baseline_file = DATA_DIR / 'climate_baselines.csv'
    baselines.to_csv(baseline_file, index=False)
    print(f"\n💾 Saved baselines to: {baseline_file.name}")
    
    # Create a JSON lookup for quick access
    baseline_dict = {}
    for _, row in baselines.iterrows():
        key = f"{row['lat_rounded']:.2f},{row['lon_rounded']:.2f},{int(row['month'])}"
        baseline_dict[key] = {
            'temperature': float(row['baseline_temp']),
            'precipitation': float(row['baseline_precip']),
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude'])
        }
    
    # Save JSON
    json_file = DATA_DIR / 'climate_baselines.json'
    with open(json_file, 'w') as f:
        json.dump(baseline_dict, f, indent=2)
    print(f"💾 Saved JSON lookup to: {json_file.name}")
    
    # Calculate and display statistics
    print("\n📈 Baseline Statistics:")
    print("-" * 60)
    print(f"Temperature Range: {baselines['baseline_temp'].min():.2f}°C to {baselines['baseline_temp'].max():.2f}°C")
    print(f"Temperature Mean: {baselines['baseline_temp'].mean():.2f}°C")
    print(f"Precipitation Range: {baselines['baseline_precip'].min():.2f}mm to {baselines['baseline_precip'].max():.2f}mm")
    print(f"Precipitation Mean: {baselines['baseline_precip'].mean():.2f}mm")
    
    # Show baselines by month (global average)
    print("\n📅 Average Baselines by Month (All Locations):")
    print("-" * 60)
    monthly_avg = baselines.groupby('month').agg({
        'baseline_temp': 'mean',
        'baseline_precip': 'mean'
    })
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for month, row in monthly_avg.iterrows():
        print(f"{month_names[int(month)-1]}: Temp={row['baseline_temp']:6.2f}°C, Precip={row['baseline_precip']:7.2f}mm")
    
    print("\n" + "=" * 60)
    print("✅ Baseline Calculation Complete!")
    print("=" * 60)
    
    return baselines


def test_baseline_lookup(lat, lon, month):
    """
    Test function to look up baseline for a specific location and month
    """
    json_file = DATA_DIR / 'climate_baselines.json'
    
    if not json_file.exists():
        print("❌ Baselines file not found. Run calculate_baselines() first.")
        return None
    
    with open(json_file, 'r') as f:
        baselines = json.load(f)
    
    # Round coordinates to match baseline data
    lat_rounded = round(lat, 2)
    lon_rounded = round(lon, 2)
    key = f"{lat_rounded:.2f},{lon_rounded:.2f},{month}"
    
    if key in baselines:
        result = baselines[key]
        print(f"\n✓ Found baseline for ({lat:.4f}°, {lon:.4f}°) - Month {month}:")
        print(f"  Temperature: {result['temperature']:.2f}°C")
        print(f"  Precipitation: {result['precipitation']:.2f}mm")
        return result
    else:
        print(f"\n❌ No baseline found for ({lat:.4f}°, {lon:.4f}°) - Month {month}")
        print(f"   Searched key: {key}")
        
        # Try to find nearest location
        print("\n🔍 Searching for nearest location...")
        closest_key = None
        min_distance = float('inf')
        
        for baseline_key in baselines.keys():
            parts = baseline_key.split(',')
            b_lat, b_lon, b_month = float(parts[0]), float(parts[1]), int(parts[2])
            
            if b_month == month:
                distance = ((lat - b_lat)**2 + (lon - b_lon)**2)**0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_key = baseline_key
        
        if closest_key:
            result = baselines[closest_key]
            parts = closest_key.split(',')
            print(f"   Nearest location: ({parts[0]}°, {parts[1]}°)")
            print(f"   Distance: {min_distance:.4f} degrees")
            print(f"   Temperature: {result['temperature']:.2f}°C")
            print(f"   Precipitation: {result['precipitation']:.2f}mm")
            return result
        
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate Climate Baselines')
    parser.add_argument('--calculate', action='store_true', help='Calculate baselines from training data')
    parser.add_argument('--test', action='store_true', help='Test baseline lookup')
    parser.add_argument('--lat', type=float, help='Latitude for test lookup')
    parser.add_argument('--lon', type=float, help='Longitude for test lookup')
    parser.add_argument('--month', type=int, help='Month (1-12) for test lookup')
    
    args = parser.parse_args()
    
    if args.calculate or (not args.test):
        # Default: calculate baselines
        baselines = calculate_baselines()
    
    if args.test:
        if args.lat is None or args.lon is None or args.month is None:
            print("❌ For testing, provide --lat, --lon, and --month")
            print("Example: python calculate_baselines.py --test --lat 28.6139 --lon 77.2090 --month 10")
        else:
            test_baseline_lookup(args.lat, args.lon, args.month)
