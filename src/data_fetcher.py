"""
NASA Climate Data Fetcher
Fetches data from NASA POWER APIs for multiple locations and time periods
"""

import requests
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))
from nasa_apis import NASAAPIs


class ClimateDataFetcher:
    """Fetch climate data from NASA POWER APIs"""
    
    def __init__(self, output_dir='data/raw'):
        self.api = NASAAPIs()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = 1  # seconds between requests to avoid overwhelming the API
    
    def fetch_temporal_monthly(self, latitude, longitude, start_year, end_year, location_id):
        """
        Fetch temporal monthly data for a location
        
        Args:
            latitude, longitude: Coordinates
            start_year, end_year: Year range
            location_id: Unique identifier for the location
            
        Returns:
            DataFrame with climate data or None if failed
        """
        try:
            url = self.api.get_temporal_monthly_url(
                longitude=longitude,
                latitude=latitude,
                start_year=start_year,
                end_year=end_year
            )
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Save raw JSON
            json_file = self.output_dir / f'temporal_monthly_{location_id}_{start_year}_{end_year}.json'
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Extract parameter data
            if 'properties' in data and 'parameter' in data['properties']:
                params_data = data['properties']['parameter']
                df = pd.DataFrame(params_data)
                
                # Add metadata
                df['latitude'] = latitude
                df['longitude'] = longitude
                df['location_id'] = location_id
                
                # Convert index (YYYYMM) to proper datetime
                # Note: NASA API includes month 13 as annual average - filter it out
                df = df.reset_index()
                df.rename(columns={'index': 'date_str'}, inplace=True)
                
                # Filter out month 13 (annual averages)
                df = df[~df['date_str'].str.endswith('13')]
                
                # Parse date as YYYYMM format
                df['time'] = pd.to_datetime(df['date_str'], format='%Y%m')
                df = df.drop(columns=['date_str'])
                
                return df
            else:
                print(f"  ⚠ No parameter data for location {location_id}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Request failed for location {location_id}: {e}")
            return None
        except Exception as e:
            print(f"  ✗ Error processing location {location_id}: {e}")
            return None
    
    def fetch_temporal_climatology(self, latitude, longitude, start_year, end_year, location_id):
        """
        Fetch temporal monthly climatology data
        Note: This endpoint may have different requirements
        
        Args:
            latitude, longitude: Coordinates
            start_year, end_year: Year range
            location_id: Unique identifier
            
        Returns:
            DataFrame or None
        """
        try:
            url = self.api.get_temporal_monthly_climatology_url(
                longitude=longitude,
                latitude=latitude,
                start_year=start_year,
                end_year=end_year
            )
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Save raw JSON
            json_file = self.output_dir / f'climatology_{location_id}_{start_year}_{end_year}.json'
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            if 'properties' in data and 'parameter' in data['properties']:
                params_data = data['properties']['parameter']
                df = pd.DataFrame(params_data)
                
                # Add metadata
                df['latitude'] = latitude
                df['longitude'] = longitude
                df['location_id'] = location_id
                
                return df
            else:
                return None
                
        except Exception as e:
            print(f"  ⚠ Climatology fetch failed for location {location_id}: {e}")
            return None
    
    def fetch_location_data(self, latitude, longitude, location_id, start_year=2005, end_year=2024):
        """
        Fetch all available data for a single location
        
        Args:
            latitude, longitude: Coordinates
            location_id: Unique identifier
            start_year, end_year: Year range
            
        Returns:
            dict with DataFrames for each API
        """
        print(f"\n  Fetching data for location {location_id} ({latitude}, {longitude})...")
        
        results = {}
        
        # Fetch temporal monthly data
        print(f"    → Temporal Monthly ({start_year}-{end_year})...")
        temporal_df = self.fetch_temporal_monthly(latitude, longitude, start_year, end_year, location_id)
        if temporal_df is not None:
            results['temporal_monthly'] = temporal_df
            print(f"      ✓ Retrieved {len(temporal_df)} months of data")
        
        # Small delay between requests
        time.sleep(self.request_delay)
        
        # Note: Climatology and Indicators APIs may need adjustment based on actual API response
        # For now, focusing on Temporal Monthly which we know works
        
        return results
    
    def fetch_multiple_locations(self, locations_df, start_year=2005, end_year=2024, max_locations=None):
        """
        Fetch data for multiple locations
        
        Args:
            locations_df: DataFrame with columns: location_id, latitude, longitude
            start_year, end_year: Year range
            max_locations: Limit number of locations (for testing)
            
        Returns:
            DataFrame with combined data from all locations
        """
        if max_locations:
            locations_df = locations_df.head(max_locations)
        
        print(f"\n{'='*70}")
        print(f"FETCHING DATA FOR {len(locations_df)} LOCATIONS")
        print(f"Time range: {start_year}-{end_year}")
        print(f"{'='*70}")
        
        all_data = []
        failed_locations = []
        
        for idx, row in tqdm(locations_df.iterrows(), total=len(locations_df), desc="Fetching locations"):
            location_id = row['location_id']
            latitude = row['latitude']
            longitude = row['longitude']
            
            results = self.fetch_location_data(
                latitude=latitude,
                longitude=longitude,
                location_id=location_id,
                start_year=start_year,
                end_year=end_year
            )
            
            if 'temporal_monthly' in results and results['temporal_monthly'] is not None:
                all_data.append(results['temporal_monthly'])
            else:
                failed_locations.append(location_id)
            
            # Delay between locations
            time.sleep(self.request_delay)
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            print(f"\n{'='*70}")
            print("SUMMARY")
            print(f"{'='*70}")
            print(f"✓ Successfully fetched: {len(all_data)} locations")
            print(f"✗ Failed: {len(failed_locations)} locations")
            print(f"Total data rows: {len(combined_df)}")
            print(f"Date range: {combined_df['time'].min()} to {combined_df['time'].max()}")
            print(f"Parameters: {[col for col in combined_df.columns if col not in ['time', 'latitude', 'longitude', 'location_id']]}")
            print(f"{'='*70}\n")
            
            return combined_df
        else:
            print("✗ No data was successfully fetched")
            return None
    
    def save_data(self, df, filename):
        """Save DataFrame to CSV"""
        if df is not None and not df.empty:
            output_path = self.output_dir.parent / filename
            df.to_csv(output_path, index=False)
            print(f"✓ Saved data to {output_path}")
            print(f"  Shape: {df.shape}")
            return output_path
        else:
            print("✗ No data to save")
            return None


def main():
    """Test the data fetcher with a few locations"""
    
    # Load location grid (you can change this to any grid file)
    locations_file = Path('data/locations_major_cities.csv')
    
    if not locations_file.exists():
        print(f"✗ Location file not found: {locations_file}")
        print("  Run location_grid.py first to generate location grids")
        return
    
    locations_df = pd.read_csv(locations_file)
    print(f"Loaded {len(locations_df)} locations from {locations_file}")
    
    # Initialize fetcher
    fetcher = ClimateDataFetcher(output_dir='data/raw')
    
    # Fetch data for first 3 locations (test run)
    print("\n⚠ Test run: Fetching data for first 3 locations only")
    combined_data = fetcher.fetch_multiple_locations(
        locations_df=locations_df,
        start_year=2020,  # Shorter period for testing
        end_year=2021,
        max_locations=3
    )
    
    # Save combined data
    if combined_data is not None:
        fetcher.save_data(combined_data, 'climate_data_test.csv')
        
        print("\nSample data:")
        print(combined_data.head(10))
        print("\nData info:")
        print(combined_data.info())


if __name__ == "__main__":
    main()
