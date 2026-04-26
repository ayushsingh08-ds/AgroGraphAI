import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.rothc import RothC, estimate_initial_pools

def run_pipeline():
    print("--- Starting RothC Pipeline (Phase 4) ---")
    
    # 1. Load Data
    data_path = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI\data\processed\clean_master_engineered.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} sites.")
    
    # Ensure rothc directory exists
    output_dir = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI\data\processed\rothc"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 4.4: Estimate Initial SOC Pools
    print("Step 4.4: Initializing Pools...")
    initial_pools_list = []
    for idx, row in df.iterrows():
        pools = estimate_initial_pools(row['soc_0_30'])
        pools['site_id'] = f"site_{idx:04d}"
        initial_pools_list.append(pools)
    
    df_initial = pd.DataFrame(initial_pools_list)
    df_initial.to_csv(os.path.join(output_dir, "initial_pools.csv"), index=False)
    
    # Step 4.5: Spin-Up (Equilibrium Run)
    print("Step 4.5: Running Spin-Up (500 years)...")
    equil_pools_list = []
    
    # Monthly columns
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    for idx, row in df.iterrows():
        # Initialize model for this site
        clay = row.get('clay_0_30', 20.0) # Default 20% clay
        model = RothC(clay_pct=clay)
        
        pools = estimate_initial_pools(row['soc_0_30'])
        
        # Spin up for 500 years
        # Use average climate for spin-up
        # (Alternatively, repeat the 12-month cycle 500 times)
        for year in range(500):
            for m in months:
                temp = row.get(f'temp_{m}', 25.0)
                rain = row.get(f'rain_{m}', 50.0)
                pet = row.get(f'pet_{m}', 100.0)
                
                # Use NDVI mean as proxy for C input
                # Following user's Step 4.2 logic
                ndvi = row.get('ndvi_mean', 0.5)
                lc = row.get('worldcover', 2)
                
                # k=1.2, LC multipliers
                k_factor = 1.2
                lc_multipliers = {1: 1.5, 2: 1.0, 3: 1.2, 4: 0.2, 5: 1.0}
                c_input = max(0, ndvi) * k_factor * lc_multipliers.get(lc, 1.0)
                
                # Cover logic Step 4.3
                is_covered = lc in [1, 2, 3, 5]
                
                pools = model.step(pools, temp, rain, pet, c_input, is_covered=is_covered)
        
        pools['site_id'] = f"site_{idx:04d}"
        equil_pools_list.append(pools)
        
    df_equil = pd.DataFrame(equil_pools_list)
    df_equil.to_csv(os.path.join(output_dir, "equilibrium_pools.csv"), index=False)
    
    # Step 4.6: Forward Simulation (2018-2025 -> 8 years)
    print("Step 4.6: Running Forward Simulation (8 years)...")
    final_pools_list = []
    
    for idx, row in df.iterrows():
        clay = row.get('clay_0_30', 20.0)
        model = RothC(clay_pct=clay)
        
        # Start from equilibrium
        pools = equil_pools_list[idx].copy()
        del pools['site_id']
        
        for year in range(8):
            for m in months:
                temp = row.get(f'temp_{m}', 25.0)
                rain = row.get(f'rain_{m}', 50.0)
                pet = row.get(f'pet_{m}', 100.0)
                ndvi = row.get('ndvi_mean', 0.5)
                lc = row.get('worldcover', 2)
                
                k_factor = 1.2
                lc_multipliers = {1: 1.5, 2: 1.0, 3: 1.2, 4: 0.2, 5: 1.0}
                c_input = max(0, ndvi) * k_factor * lc_multipliers.get(lc, 1.0)
                is_covered = lc in [1, 2, 3, 5]
                
                pools = model.step(pools, temp, rain, pet, c_input, is_covered=is_covered)
        
        pools['site_id'] = f"site_{idx:04d}"
        final_pools_list.append(pools)
        
    df_final_pools = pd.DataFrame(final_pools_list)
    df_final_pools.to_csv(os.path.join(output_dir, "final_soc.csv"), index=False)
    
    # Step 4.7: Extract Final SOC
    print("Step 4.7: Extracting Final SOC...")
    df_results = df[['x', 'y', 'soc_0_30']].copy()
    df_results.rename(columns={'soc_0_30': 'soc_observed'}, inplace=True)
    
    # Sum pools
    df_results['soc_rothc'] = df_final_pools[['DPM', 'RPM', 'BIO', 'HUM', 'IOM']].sum(axis=1)
    
    # Step 4.8: Compute Residual
    print("Step 4.8: Computing Residuals...")
    df_results['residual_soc'] = df_results['soc_observed'] - df_results['soc_rothc']
    
    df_results['site_id'] = [f"site_{i:04d}" for i in range(len(df_results))]
    df_results.to_csv(os.path.join(output_dir, "residuals.csv"), index=False)
    
    # Step 4.9: Validate RothC Baseline
    print("Step 4.9: Validating Baseline...")
    y_true = df_results['soc_observed']
    y_pred = df_results['soc_rothc']
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'Bias': np.mean(y_pred - y_true),
        'R2': r2_score(y_true, y_pred)
    }
    
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    
    print("\n--- Pipeline Complete ---")
    print(metrics)

if __name__ == "__main__":
    run_pipeline()
