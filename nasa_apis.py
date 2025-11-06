"""
NASA API Configuration for Climate Trends Prediction
This file contains API endpoints and configurations for fetching climate data
"""

class NASAAPIs:
    """Class to manage NASA API endpoints and parameters"""
    
    BASE_URL = "https://power.larc.nasa.gov"
    
    # API 1: Temporal Monthly NASA POWER API
    TEMPORAL_MONTHLY = {
        "endpoint": "/api/temporal/monthly/point",
        "description": "Monthly temporal data for specific location",
        "selected_parameters": [
            # Core (must-have for trend + extremes)
            "T2M",          # Temperature at 2 Meters - mean (use for anomalies & trend targets)
            "T2M_MAX",      # Maximum Temperature at 2 Meters (extreme heat signal)
            "T2M_MIN",      # Minimum Temperature at 2 Meters (nighttime cooling trends, frost risk)
            
            # Hydrology / drought (important for climate impact)
            "PRECTOTCORR",  # Corrected precipitation total (monthly total precipitation)
            
            # Radiation / energy balance (helps explain temp changes)
            "ALLSKY_SFC_SW_DWN",  # Surface shortwave downwards (incoming solar radiation)
            
            # Atmospheric moisture & heat stress
            "RH2M",         # Relative Humidity at 2 Meters (needed for heat-index / wet-bulb estimates)
            "QV2M"          # Specific Humidity at 2 Meters (human impacts)
        ],
        "default_params": {
            "community": "SB",
            "format": "JSON"
        },
        "example_url": "/api/temporal/monthly/point?parameters=T2M,T2M_MAX&community=SB&longitude=0&latitude=0&format=JSON&start=2016&end=2017"
    }
    
    @staticmethod
    def get_temporal_monthly_url(longitude, latitude, start_year, end_year, parameters=None):
        """
        Generate URL for temporal monthly data
        
        Args:
            longitude (float): Longitude coordinate
            latitude (float): Latitude coordinate
            start_year (int): Start year for data
            end_year (int): End year for data
            parameters (list): List of parameters to fetch (default: T2M, T2M_MAX)
            
        Returns:
            str: Complete API URL
        """
        if parameters is None:
            parameters = NASAAPIs.TEMPORAL_MONTHLY["selected_parameters"]
        
        params_str = ",".join(parameters)
        url = (f"{NASAAPIs.BASE_URL}{NASAAPIs.TEMPORAL_MONTHLY['endpoint']}"
               f"?parameters={params_str}"
               f"&community={NASAAPIs.TEMPORAL_MONTHLY['default_params']['community']}"
               f"&longitude={longitude}"
               f"&latitude={latitude}"
               f"&format={NASAAPIs.TEMPORAL_MONTHLY['default_params']['format']}"
               f"&start={start_year}"
               f"&end={end_year}")
        return url
    
    # API 2: Temporal Monthly Climatology NASA POWER API
    TEMPORAL_MONTHLY_CLIMATOLOGY = {
        "endpoint": "/api/temporal/monthly/climatology/point",
        "description": "Monthly climatology data for specific location",
        "selected_parameters": [
            # Core (must-have for trend + extremes)
            "T2M",          # Temperature at 2 Meters - mean (use for anomalies & trend targets)
            "T2M_MAX",      # Maximum Temperature at 2 Meters (extreme heat signal)
            "T2M_MIN",      # Minimum Temperature at 2 Meters (nighttime cooling trends, frost risk)
            
            # Hydrology / drought (important for climate impact)
            "PRECTOTCORR",  # Corrected precipitation total (monthly total precipitation)
            
            # Radiation / energy balance (helps explain temp changes)
            "ALLSKY_SFC_SW_DWN",  # Surface shortwave downwards (incoming solar radiation)
            
            # Atmospheric moisture & heat stress
            "RH2M",         # Relative Humidity at 2 Meters (needed for heat-index / wet-bulb estimates)
            "QV2M"          # Specific Humidity at 2 Meters (human impacts)
        ],
        "default_params": {
            "community": "SB",
            "format": "JSON"
        },
        "example_url": "/api/temporal/monthly/climatology/point?parameters=T2M,T2M_MAX&community=SB&longitude=0&latitude=0&format=JSON&start=2016&end=2017"
    }
    
    @staticmethod
    def get_temporal_monthly_climatology_url(longitude, latitude, start_year, end_year, parameters=None):
        """
        Generate URL for temporal monthly climatology data
        
        Args:
            longitude (float): Longitude coordinate
            latitude (float): Latitude coordinate
            start_year (int): Start year for data
            end_year (int): End year for data
            parameters (list): List of parameters to fetch (default: T2M, T2M_MAX)
            
        Returns:
            str: Complete API URL
        """
        if parameters is None:
            parameters = NASAAPIs.TEMPORAL_MONTHLY_CLIMATOLOGY["selected_parameters"]
        
        params_str = ",".join(parameters)
        url = (f"{NASAAPIs.BASE_URL}{NASAAPIs.TEMPORAL_MONTHLY_CLIMATOLOGY['endpoint']}"
               f"?parameters={params_str}"
               f"&community={NASAAPIs.TEMPORAL_MONTHLY_CLIMATOLOGY['default_params']['community']}"
               f"&longitude={longitude}"
               f"&latitude={latitude}"
               f"&format={NASAAPIs.TEMPORAL_MONTHLY_CLIMATOLOGY['default_params']['format']}"
               f"&start={start_year}"
               f"&end={end_year}")
        return url
    
    # API 3: Climate Indicators API
    CLIMATE_INDICATORS = {
        "endpoint": "/api/application/indicators/point",
        "description": "Climate indicators data for specific location",
        "selected_parameters": [
            # Extreme temperature indices
            "TXx",          # Highest daily maximum temperature (extreme warm index)
            "TNn",          # Lowest daily minimum temperature (extreme cold index)
            
            # Drought/precipitation indices
            "CDD",          # Consecutive Dry Days (drought signal)
            "CWD",          # Consecutive Wet Days
            "SPI",          # Standardized Precipitation Index (or SPI-like indices)
            
            # Degree days (energy/agriculture impact)
            "HDD",          # Heating Degree Days (impact on heating energy needs)
            "CDD_energy",   # Cooling Degree Days (impact on cooling energy needs)
            
            # Heat stress indicators
            "hot_days",     # Count of hot days (e.g., days > 35°C) - impact risk mapping
            
            # Agricultural indices
            "frost_days",   # Frost days (days with minimum temp below 0°C)
            "GSL",          # Growing Season Length
            "GDD",          # Growing Degree Days (agricultural productivity)
        ],
        "default_params": {
            "format": "JSON"  # Changed from HTML to JSON for easier data processing
        },
        "example_url": "/api/application/indicators/point?longitude=-84.43&latitude=33.64&start=2001&end=2016&format=HTML",
        "note": "This API returns predefined climate indicators. Exact indicator names may vary - inspect API response to confirm available indicators."
    }
    
    @staticmethod
    def get_climate_indicators_url(longitude, latitude, start_year, end_year, format_type="JSON"):
        """
        Generate URL for climate indicators data
        
        Args:
            longitude (float): Longitude coordinate
            latitude (float): Latitude coordinate
            start_year (int): Start year for data
            end_year (int): End year for data
            format_type (str): Output format (JSON or HTML, default: JSON)
            
        Returns:
            str: Complete API URL
        """
        url = (f"{NASAAPIs.BASE_URL}{NASAAPIs.CLIMATE_INDICATORS['endpoint']}"
               f"?longitude={longitude}"
               f"&latitude={latitude}"
               f"&start={start_year}"
               f"&end={end_year}"
               f"&format={format_type}")
        return url
    
    # Placeholder for additional APIs
    # Will be added as you share more APIs


if __name__ == "__main__":
    # Example usage
    api = NASAAPIs()
    
    # Generate example URLs
    print("="*60)
    print("API 1: Temporal Monthly")
    print("="*60)
    url1 = api.get_temporal_monthly_url(
        longitude=0,
        latitude=0,
        start_year=2016,
        end_year=2017
    )
    print(f"Description: {NASAAPIs.TEMPORAL_MONTHLY['description']}")
    print(f"Selected Parameters: {', '.join(NASAAPIs.TEMPORAL_MONTHLY['selected_parameters'])}")
    print(f"Example URL:\n{url1}")
    
    print("\n" + "="*60)
    print("API 2: Temporal Monthly Climatology")
    print("="*60)
    url2 = api.get_temporal_monthly_climatology_url(
        longitude=0,
        latitude=0,
        start_year=2016,
        end_year=2017
    )
    print(f"Description: {NASAAPIs.TEMPORAL_MONTHLY_CLIMATOLOGY['description']}")
    print(f"Selected Parameters: {', '.join(NASAAPIs.TEMPORAL_MONTHLY_CLIMATOLOGY['selected_parameters'])}")
    print(f"Example URL:\n{url2}")
    
    print("\n" + "="*60)
    print("API 3: Climate Indicators")
    print("="*60)
    url3 = api.get_climate_indicators_url(
        longitude=-84.43,
        latitude=33.64,
        start_year=2001,
        end_year=2016
    )
    print(f"Description: {NASAAPIs.CLIMATE_INDICATORS['description']}")
    print(f"Note: Returns predefined climate indicators (no parameter selection needed)")
    print(f"Example URL:\n{url3}")
