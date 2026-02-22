import os
import json
import pandas as pd
import glob
import sys
import numpy as np

def extract_outlier_filenames(results_dir):
    """
    Reads all worker JSON files in the given directory, computes the global 
    'Best Result' (minimum layers per instance), calculates the 2SD layer outlier 
    threshold, and prints a newline-separated list of the raw CNF filenames.
    """
    # 1. Find all worker files exactly like the plotting script
    search_pattern = os.path.join(results_dir, "benchmark_*_worker*_all_*.json")
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        print(f"Error: No worker JSON files found in {results_dir}", file=sys.stderr)
        sys.exit(1)
        
    # 2. Load and concatenate all data
    all_records = []
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_records.extend(data)
                elif isinstance(data, dict) and "results" in data:
                    all_records.extend(data["results"])
                else:
                    print(f"Warning: Unexpected JSON structure in {file_path}", file=sys.stderr)
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)
            
    df = pd.DataFrame(all_records)
    
    if df.empty:
        print("Error: DataFrame is empty. No valid data extracted.", file=sys.stderr)
        sys.exit(1)
        
    if 'filename' not in df.columns or 'layers' not in df.columns:
        print("Error: Traces missing required 'filename' or 'layers' columns.", file=sys.stderr)
        sys.exit(1)
        
    # 3. Compute "Best Result" per instance (minimum layers)
    # The plotting script defines 'best' logically as the run with the minimum energy,
    # or the minimum layers if tied. We will group by filename and take the one 
    # with the absolute minimum layers achieved across any configuration.
    df_best = df.loc[df.groupby('filename')['layers'].idxmin()]
    
    # 4. Calculate 2SD layer threshold on the Best Result dataset
    layer_median = df_best['layers'].median()
    layer_std = df_best['layers'].std()
    
    # Handle edge case where std might be 0 or NaN
    if pd.isna(layer_std):
        layer_std = 0.0
        
    layer_threshold = layer_median + (2 * layer_std)
    
    # 5. Extract outliers
    outliers_df = df_best[df_best['layers'] > layer_threshold]
    
    # Extract just the raw basenames
    outlier_files = outliers_df['filename'].apply(os.path.basename).unique()
    
    # 6. Print to stdout (for bash pipe)
    # We print ONLY the filenames to stdout so bash can read them cleanly.
    # We print log info to stderr so it shows up on console but doesn't break pipes.
    print(f"Extracted {len(outlier_files)} outliers (Threshold: >{layer_threshold:.2f} layers) from {len(df_best)} total instances.", file=sys.stderr)
    
    for filename in outlier_files:
        print(filename)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_outliers.py <path_to_results_directory>", file=sys.stderr)
        sys.exit(1)
        
    results_target = sys.argv[1]
    if not os.path.isdir(results_target):
        print(f"Error: {results_target} is not a valid directory.", file=sys.stderr)
        sys.exit(1)
        
    extract_outlier_filenames(results_target)
