"""
Main Data Collection Pipeline
Orchestrates the complete data collection and processing workflow

Usage:
    python main_pipeline.py --grid india --start 2010 --end 2024
    python main_pipeline.py --grid global --start 2005 --end 2024 --test
    python main_pipeline.py --grid cities --start 2015 --end 2024
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.location_grid import LocationGrid
from src.data_fetcher import ClimateDataFetcher
from src.data_processor import DataProcessor


def run_pipeline(grid_type='cities', start_year=2010, end_year=2024, 
                 test_mode=False, max_locations=None):
    """
    Run complete data collection pipeline
    
    Args:
        grid_type: Type of location grid ('global', 'india', 'cities')
        start_year: Start year for data collection
        end_year: End year for data collection
        test_mode: If True, only fetch data for a few locations
        max_locations: Maximum number of locations to fetch (None = all)
    """
    
    print("\n" + "="*70)
    print("NASA CLIMATE DATA COLLECTION PIPELINE")
    print("="*70)
    print(f"Grid Type: {grid_type}")
    print(f"Time Period: {start_year}-{end_year}")
    print(f"Test Mode: {test_mode}")
    if max_locations:
        print(f"Max Locations: {max_locations}")
    print("="*70 + "\n")
    
    # Step 1: Generate location grid
    print("STEP 1: GENERATING LOCATION GRID")
    print("-" * 70)
    
    if grid_type == 'global':
        locations_df = LocationGrid.generate_global_grid(lat_step=10, lon_step=10)
        grid_filename = 'locations_global_10deg.csv'
    elif grid_type == 'india':
        locations_df = LocationGrid.generate_india_grid(step=2)
        grid_filename = 'locations_india_2deg.csv'
    elif grid_type == 'cities':
        locations_df = LocationGrid.load_city_coordinates()
        grid_filename = 'locations_major_cities.csv'
    else:
        print(f"✗ Unknown grid type: {grid_type}")
        return
    
    # Save location grid
    LocationGrid.save_grid(locations_df, grid_filename)
    locations_file = Path('data') / grid_filename
    
    # In test mode, limit locations
    if test_mode:
        max_locations = min(3, len(locations_df))
        print(f"\n⚠ TEST MODE: Limiting to {max_locations} locations")
    
    # Step 2: Fetch climate data
    print("\n" + "="*70)
    print("STEP 2: FETCHING CLIMATE DATA FROM NASA POWER APIs")
    print("-" * 70)
    
    fetcher = ClimateDataFetcher(output_dir='data/raw')
    
    combined_data = fetcher.fetch_multiple_locations(
        locations_df=locations_df,
        start_year=start_year,
        end_year=end_year,
        max_locations=max_locations
    )
    
    if combined_data is None or combined_data.empty:
        print("✗ Failed to fetch any data. Aborting pipeline.")
        return
    
    # Save raw combined data
    raw_data_file = f'climate_data_{grid_type}_{start_year}_{end_year}.csv'
    fetcher.save_data(combined_data, raw_data_file)
    
    # Step 3: Process and create master dataset
    print("\n" + "="*70)
    print("STEP 3: PROCESSING DATA AND CREATING MASTER DATASET")
    print("-" * 70)
    
    processor = DataProcessor(data_dir='data')
    
    master_df = processor.create_master_dataset(
        climate_data_file=Path('data') / raw_data_file,
        locations_file=locations_file,
        add_anomalies=False,  # Set to True if you have baseline period data
        add_derived=True,
        add_lags=False  # Set to True for time-series models
    )
    
    # Save master dataset
    master_filename = f'climate_master_{grid_type}_{start_year}_{end_year}.csv'
    processor.save_master_dataset(master_df, master_filename)
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nGenerated files in 'data/' directory:")
    print(f"  1. {grid_filename} - Location grid")
    print(f"  2. {raw_data_file} - Raw climate data")
    print(f"  3. {master_filename} - Processed master dataset")
    print(f"\nMaster dataset shape: {master_df.shape}")
    print(f"  Rows: {len(master_df)}")
    print(f"  Columns: {len(master_df.columns)}")
    print(f"  Locations: {master_df['location_id'].nunique()}")
    print(f"  Time range: {master_df['time'].min()} to {master_df['time'].max()}")
    print("="*70 + "\n")
    
    # Display sample
    print("Sample data (first 5 rows):")
    print(master_df.head())
    print("\n" + "="*70)
    
    return master_df


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='NASA Climate Data Collection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch data for major cities (2010-2024)
  python main_pipeline.py --grid cities --start 2010 --end 2024
  
  # Test run with India grid (only 3 locations)
  python main_pipeline.py --grid india --start 2020 --end 2021 --test
  
  # Global grid (10° resolution)
  python main_pipeline.py --grid global --start 2005 --end 2024
  
  # Limit number of locations
  python main_pipeline.py --grid cities --start 2015 --end 2024 --max-locations 5
        """
    )
    
    parser.add_argument('--grid', type=str, default='cities',
                       choices=['global', 'india', 'cities'],
                       help='Type of location grid (default: cities)')
    
    parser.add_argument('--start', type=int, default=2010,
                       help='Start year for data collection (default: 2010)')
    
    parser.add_argument('--end', type=int, default=2024,
                       help='End year for data collection (default: 2024)')
    
    parser.add_argument('--test', action='store_true',
                       help='Test mode: only fetch 3 locations')
    
    parser.add_argument('--max-locations', type=int, default=None,
                       help='Maximum number of locations to fetch (optional)')
    
    args = parser.parse_args()
    
    # Run pipeline
    run_pipeline(
        grid_type=args.grid,
        start_year=args.start,
        end_year=args.end,
        test_mode=args.test,
        max_locations=args.max_locations
    )


if __name__ == "__main__":
    main()
