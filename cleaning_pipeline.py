"""
Data Cleaning Pipeline Script
Run comprehensive data cleaning on the master dataset

Usage:
    python cleaning_pipeline.py
    python cleaning_pipeline.py --input data/climate_master_cities_2010_2024.csv
    python cleaning_pipeline.py --scale-method robust --outlier-method winsorize
"""

import argparse
from pathlib import Path
import pandas as pd
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data_cleaner import DataCleaner


def run_cleaning_pipeline(input_file, output_file=None, scale_method='standard',
                          outlier_method='clip', handle_outliers=True):
    """
    Run comprehensive data cleaning pipeline
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output cleaned CSV file (auto-generated if None)
        scale_method: Scaling method ('standard', 'minmax', 'robust')
        outlier_method: Outlier treatment ('clip', 'remove', 'winsorize')
        handle_outliers: Whether to detect and treat outliers
    """
    
    print("\n" + "="*80)
    print("DATA CLEANING PIPELINE")
    print("="*80)
    print(f"Input file: {input_file}")
    print(f"Scale method: {scale_method}")
    print(f"Outlier method: {outlier_method}")
    print(f"Handle outliers: {handle_outliers}")
    print("="*80 + "\n")
    
    # Check if input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"✗ Input file not found: {input_file}")
        return None
    
    # Load data
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns\n")
    
    # Initialize cleaner
    cleaner = DataCleaner(data_dir='data', models_dir='models')
    
    # Run cleaning pipeline
    cleaned_df = cleaner.clean_pipeline(
        df=df,
        for_training=True,
        scale_method=scale_method,
        handle_outliers=handle_outliers,
        outlier_method=outlier_method
    )
    
    # Generate output filename if not provided
    if output_file is None:
        input_stem = input_path.stem
        # Replace 'master' with 'cleaned' in filename
        if 'master' in input_stem:
            output_stem = input_stem.replace('master', 'cleaned')
        else:
            output_stem = input_stem + '_cleaned'
        output_file = input_path.parent / f"{output_stem}.csv"
    else:
        output_file = Path(output_file)
    
    # Save cleaned data
    cleaned_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved cleaned data to {output_file}")
    print(f"  Final shape: {cleaned_df.shape}")
    
    # Save summary statistics
    summary_file = output_file.parent / f"{output_file.stem}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CLEANED DATASET SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Output file: {output_file}\n\n")
        f.write(f"Shape: {cleaned_df.shape}\n")
        f.write(f"  Rows: {len(cleaned_df):,}\n")
        f.write(f"  Columns: {len(cleaned_df.columns)}\n\n")
        f.write("Column Names:\n")
        for i, col in enumerate(cleaned_df.columns, 1):
            f.write(f"  {i:2d}. {col}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("STATISTICAL SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(str(cleaned_df.describe()))
        f.write("\n\n" + "="*80 + "\n")
        f.write("DATA TYPES\n")
        f.write("="*80 + "\n")
        f.write(str(cleaned_df.dtypes))
        f.write("\n")
    
    print(f"✓ Saved summary to {summary_file}")
    
    # Display sample
    print("\n" + "="*80)
    print("SAMPLE CLEANED DATA (First 5 rows)")
    print("="*80)
    print(cleaned_df.head())
    
    print("\n" + "="*80)
    print("CLEANING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  1. {output_file} - Cleaned dataset")
    print(f"  2. {summary_file} - Summary statistics")
    print(f"  3. models/cleaning_scalers.pkl - Fitted scalers")
    print(f"  4. models/cleaning_encoders.pkl - Fitted encoders")
    print(f"  5. models/cleaning_report.json - Cleaning report")
    print("="*80 + "\n")
    
    return cleaned_df


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Data Cleaning Pipeline for NASA Climate Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean the default master dataset
  python cleaning_pipeline.py
  
  # Clean a specific file
  python cleaning_pipeline.py --input data/climate_master_cities_2010_2024.csv
  
  # Use robust scaling (better for outliers)
  python cleaning_pipeline.py --scale-method robust
  
  # Remove outliers instead of clipping
  python cleaning_pipeline.py --outlier-method remove
  
  # Use MinMax scaling (scales to 0-1 range)
  python cleaning_pipeline.py --scale-method minmax --outlier-method winsorize
  
  # Skip outlier detection
  python cleaning_pipeline.py --no-handle-outliers
        """
    )
    
    parser.add_argument('--input', type=str, 
                       default='data/climate_master_cities_2010_2024.csv',
                       help='Input CSV file path (default: data/climate_master_cities_2010_2024.csv)')
    
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (auto-generated if not provided)')
    
    parser.add_argument('--scale-method', type=str, default='standard',
                       choices=['standard', 'minmax', 'robust'],
                       help='Scaling method (default: standard)')
    
    parser.add_argument('--outlier-method', type=str, default='clip',
                       choices=['clip', 'remove', 'winsorize', 'log_transform'],
                       help='Outlier treatment method (default: clip)')
    
    parser.add_argument('--no-handle-outliers', action='store_true',
                       help='Skip outlier detection and treatment')
    
    args = parser.parse_args()
    
    # Run cleaning pipeline
    run_cleaning_pipeline(
        input_file=args.input,
        output_file=args.output,
        scale_method=args.scale_method,
        outlier_method=args.outlier_method,
        handle_outliers=not args.no_handle_outliers
    )


if __name__ == "__main__":
    main()
