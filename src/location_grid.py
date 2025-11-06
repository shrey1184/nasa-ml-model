"""
Location Grid Generator for Climate Data Collection
Generates latitude/longitude coordinates for data fetching
"""

import numpy as np
import pandas as pd
from pathlib import Path


class LocationGrid:
    """Generate and manage location grids for climate data collection"""
    
    @staticmethod
    def generate_global_grid(lat_step=10, lon_step=10):
        """
        Generate global grid of coordinates
        
        Args:
            lat_step: Latitude step in degrees
            lon_step: Longitude step in degrees
            
        Returns:
            DataFrame with lat, lon, location_id
        """
        latitudes = np.arange(-90, 91, lat_step)
        longitudes = np.arange(-180, 181, lon_step)
        
        locations = []
        location_id = 0
        
        for lat in latitudes:
            for lon in longitudes:
                locations.append({
                    'location_id': location_id,
                    'latitude': lat,
                    'longitude': lon,
                    'grid_type': 'global',
                    'description': f'Global_{lat}_{lon}'
                })
                location_id += 1
        
        return pd.DataFrame(locations)
    
    @staticmethod
    def generate_regional_grid(lat_min, lat_max, lon_min, lon_max, step=2):
        """
        Generate regional grid (e.g., India, USA, etc.)
        
        Args:
            lat_min, lat_max: Latitude range
            lon_min, lon_max: Longitude range
            step: Grid spacing in degrees
            
        Returns:
            DataFrame with lat, lon, location_id
        """
        latitudes = np.arange(lat_min, lat_max + step, step)
        longitudes = np.arange(lon_min, lon_max + step, step)
        
        locations = []
        location_id = 0
        
        for lat in latitudes:
            for lon in longitudes:
                locations.append({
                    'location_id': location_id,
                    'latitude': lat,
                    'longitude': lon,
                    'grid_type': 'regional',
                    'description': f'Regional_{lat}_{lon}'
                })
                location_id += 1
        
        return pd.DataFrame(locations)
    
    @staticmethod
    def generate_india_grid(step=2):
        """
        Generate grid covering India
        Approximate bounds: 8°N to 37°N, 68°E to 97°E
        
        Args:
            step: Grid spacing in degrees
            
        Returns:
            DataFrame with lat, lon, location_id
        """
        return LocationGrid.generate_regional_grid(
            lat_min=8, lat_max=37,
            lon_min=68, lon_max=97,
            step=step
        )
    
    @staticmethod
    def load_city_coordinates(csv_file=None):
        """
        Load major city coordinates from CSV file
        
        Expected CSV format:
        city_name, latitude, longitude, country
        
        Args:
            csv_file: Path to CSV file with city coordinates
            
        Returns:
            DataFrame with lat, lon, location_id, city_name
        """
        if csv_file and Path(csv_file).exists():
            df = pd.read_csv(csv_file)
            df['location_id'] = range(len(df))
            df['grid_type'] = 'city'
            df['description'] = df['city_name']
            return df[['location_id', 'latitude', 'longitude', 'grid_type', 'description']]
        else:
            # Return sample major cities if no file provided
            cities = [
                {'city_name': 'New Delhi', 'latitude': 28.61, 'longitude': 77.23, 'country': 'India'},
                {'city_name': 'Mumbai', 'latitude': 19.08, 'longitude': 72.88, 'country': 'India'},
                {'city_name': 'Bangalore', 'latitude': 12.97, 'longitude': 77.59, 'country': 'India'},
                {'city_name': 'Chennai', 'latitude': 13.08, 'longitude': 80.27, 'country': 'India'},
                {'city_name': 'Kolkata', 'latitude': 22.57, 'longitude': 88.36, 'country': 'India'},
                {'city_name': 'New York', 'latitude': 40.71, 'longitude': -74.01, 'country': 'USA'},
                {'city_name': 'Los Angeles', 'latitude': 34.05, 'longitude': -118.24, 'country': 'USA'},
                {'city_name': 'London', 'latitude': 51.51, 'longitude': -0.13, 'country': 'UK'},
                {'city_name': 'Tokyo', 'latitude': 35.68, 'longitude': 139.69, 'country': 'Japan'},
                {'city_name': 'Sydney', 'latitude': -33.87, 'longitude': 151.21, 'country': 'Australia'},
            ]
            df = pd.DataFrame(cities)
            df['location_id'] = range(len(df))
            df['grid_type'] = 'city'
            df['description'] = df['city_name']
            return df[['location_id', 'latitude', 'longitude', 'grid_type', 'description']]
    
    @staticmethod
    def save_grid(grid_df, filename):
        """Save location grid to CSV"""
        output_path = Path('data') / filename
        output_path.parent.mkdir(exist_ok=True)
        grid_df.to_csv(output_path, index=False)
        print(f"✓ Saved {len(grid_df)} locations to {output_path}")
        return output_path


def main():
    """Generate sample location grids"""
    
    print("\n" + "="*70)
    print("LOCATION GRID GENERATION")
    print("="*70)
    
    # Option 1: Global grid (10° x 10°)
    print("\n1. Generating Global Grid (10° x 10°)...")
    global_grid = LocationGrid.generate_global_grid(lat_step=10, lon_step=10)
    LocationGrid.save_grid(global_grid, 'locations_global_10deg.csv')
    
    # Option 2: India regional grid (2° x 2°)
    print("\n2. Generating India Regional Grid (2° x 2°)...")
    india_grid = LocationGrid.generate_india_grid(step=2)
    LocationGrid.save_grid(india_grid, 'locations_india_2deg.csv')
    
    # Option 3: Major cities
    print("\n3. Generating Major Cities Grid...")
    cities_grid = LocationGrid.load_city_coordinates()
    LocationGrid.save_grid(cities_grid, 'locations_major_cities.csv')
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Global Grid (10°):     {len(global_grid)} locations")
    print(f"India Grid (2°):       {len(india_grid)} locations")
    print(f"Major Cities:          {len(cities_grid)} locations")
    print("="*70 + "\n")
    
    # Display sample locations
    print("Sample locations from each grid:")
    print("\nGlobal Grid (first 5):")
    print(global_grid.head())
    print("\nIndia Grid (first 5):")
    print(india_grid.head())
    print("\nMajor Cities:")
    print(cities_grid)


if __name__ == "__main__":
    main()
