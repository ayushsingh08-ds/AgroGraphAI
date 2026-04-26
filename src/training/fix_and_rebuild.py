import os
import glob
import numpy as np
import pandas as pd
import rasterio
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def fix_and_rebuild():
    print("--- Starting Fix and Rebuild ---")
    
    base_dir = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI"
    source_dir = os.path.join(base_dir, "data", "processed", "standardized")
    output_dir = os.path.join(base_dir, "data", "processed")
    
    # 1. Re-extract truly unscaled data
    print("1. Re-extracting unscaled tabular data from standardized rasters...")
    all_files = glob.glob(os.path.join(source_dir, '*.tif'))
    data_dict = {}
    valid_mask = None
    reference_shape = None
    
    for filepath in all_files:
        col_name = os.path.basename(filepath).replace('.tif', '')
        with rasterio.open(filepath) as src:
            arr = src.read(1).astype(np.float32)
            if reference_shape is None:
                reference_shape = arr.shape
            
            if arr.shape != reference_shape:
                print(f"Skipping {col_name} due to shape mismatch: {arr.shape} vs {reference_shape}")
                continue

            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            
            # Temporary storage to update mask after first pass if needed
            data_dict[col_name] = arr
            
            if valid_mask is None:
                valid_mask = ~np.isnan(arr)
            else:
                valid_mask &= ~np.isnan(arr)
                
    # Now extract only valid pixels from all stored arrays
    for col_name in data_dict:
        data_dict[col_name] = data_dict[col_name][valid_mask]
            
    df = pd.DataFrame(data_dict)
    
    # Add coordinates (x, y)
    with rasterio.open(all_files[0]) as src:
        rows, cols = np.where(valid_mask)
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        df['x'] = xs
        df['y'] = ys
        
    # Save truly clean master
    clean_csv = os.path.join(output_dir, "clean_master.csv")
    df.to_csv(clean_csv, index=False)
    print(f"Saved unscaled baseline: {len(df)} samples.")
    
    # 2. Re-run Feature Engineering (Simplified for fix)
    print("2. Re-running Feature Engineering...")
    # 0-30cm weighting
    for prop in ['soc', 'clay', 'sand', 'silt', 'ph', 'bd', 'cec']:
        cols = [f'{prop}_0-5', f'{prop}_5-15', f'{prop}_15-30']
        if all(c in df.columns for c in cols):
            df[f'{prop}_0_30'] = (df[cols[0]]*5 + df[cols[1]]*10 + df[cols[2]]*15) / 30.0
            
    # Terrain features
    if 'dem' in df.columns and 'slope' in df.columns:
        df['twi'] = np.log(df['ndvi'].clip(lower=0.1) / (df['slope'].clip(lower=0.01))) # Proxy TWI
        
    # Climate aggregation
    rain_cols = [c for c in df.columns if 'rain_' in c]
    if rain_cols:
        df['annual_rain'] = df[rain_cols].sum(axis=1)
        
    df.to_csv(os.path.join(output_dir, "clean_master_engineered.csv"), index=False)
    
    # 3. Re-run RothC Pipeline
    print("3. Re-running RothC Pipeline...")
    from src.run_rothc_pipeline import run_pipeline
    run_pipeline()
    
    # 4. Re-run Graph Construction
    print("4. Re-running Graph Construction...")
    from src.build_graph import build_graph_pipeline
    build_graph_pipeline()
    
    # 5. Re-run Audit
    print("5. Re-running Audit...")
    from src.audit_data import audit_data
    audit_data()

if __name__ == "__main__":
    fix_and_rebuild()
