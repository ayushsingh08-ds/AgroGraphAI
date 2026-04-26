import os
import glob
import numpy as np
import pandas as pd
import rasterio
import json
from sklearn.preprocessing import StandardScaler
import pickle

def finalize_mapping():
    print("--- Phase 9C: Final SOC Mapping ---")
    
    base_dir = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI"
    source_dir = os.path.join(base_dir, "data", "processed", "standardized")
    output_dir = os.path.join(base_dir, "data", "results", "maps")
    eval_dir = os.path.join(base_dir, "data", "processed", "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Uncertainty Predictions
    uncer_df = pd.read_csv(os.path.join(eval_dir, "uncertainty_predictions.csv"))
    res_df = pd.read_csv(os.path.join(base_dir, "data", "processed", "rothc", "residuals.csv"))
    
    # 2. Get Georeferencing
    ref_file = glob.glob(os.path.join(source_dir, "*.tif"))[0]
    with rasterio.open(ref_file) as src:
        meta = src.meta.copy()
        meta.update(dtype=rasterio.float32, count=1, nodata=-9999)
        arr_ref = src.read(1)
        valid_mask = (arr_ref != src.nodata) & (~np.isnan(arr_ref))
        rows, cols = np.where(valid_mask)
        
        # 3. Create Map Arrays
        soc_final = np.full(src.shape, -9999, dtype=np.float32)
        uncertainty = np.full(src.shape, -9999, dtype=np.float32)
        error = np.full(src.shape, -9999, dtype=np.float32)
        
        # Calculate Final SOC (RothC + Predicted Residual)
        # We'll use the MC Mean from uncertainty_predictions
        soc_val = res_df['soc_rothc'].values + uncer_df['pred_mean'].values
        
        # Calculate Error (Observed - Predicted)
        error_val = res_df['soc_observed'].values - soc_val
        
        # Map to grid
        # Ensure we only use the number of samples we have
        num_s = len(soc_val)
        soc_final[rows[:num_s], cols[:num_s]] = soc_val
        uncertainty[rows[:num_s], cols[:num_s]] = uncer_df['pred_std'].values
        error[rows[:num_s], cols[:num_s]] = error_val
        
        # 4. Save TIFs
        with rasterio.open(os.path.join(output_dir, "soc_final_map.tif"), 'w', **meta) as dst:
            dst.write(soc_final, 1)
        with rasterio.open(os.path.join(output_dir, "soc_uncertainty_map.tif"), 'w', **meta) as dst:
            dst.write(uncertainty, 1)
        with rasterio.open(os.path.join(output_dir, "soc_error_map.tif"), 'w', **meta) as dst:
            dst.write(error, 1)

    # 5. Hotspot Detection (Top 10%)
    print("Step 9.15: Detecting Hotspots...")
    q90_soc = np.percentile(soc_val, 90)
    q90_uncer = np.percentile(uncer_df['pred_std'].values, 90)
    
    hotspots = []
    for i in range(num_s):
        if soc_val[i] >= q90_soc:
            hotspots.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(res_df['x'].iloc[i]), float(res_df['y'].iloc[i])]},
                "properties": {"type": "High SOC", "value": float(soc_val[i])}
            })
        if uncer_df['pred_std'].iloc[i] >= q90_uncer:
            hotspots.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(res_df['x'].iloc[i]), float(res_df['y'].iloc[i])]},
                "properties": {"type": "High Uncertainty", "value": float(uncer_df['pred_std'].iloc[i])}
            })
            
    with open(os.path.join(output_dir, "hotspots.geojson"), 'w') as f:
        json.dump({"type": "FeatureCollection", "features": hotspots}, f)

    print("Final Mapping Complete. Results in data/results/maps/")

if __name__ == "__main__":
    finalize_mapping()
