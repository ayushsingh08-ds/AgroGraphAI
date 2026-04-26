import os
import shutil
import pandas as pd
import json

def finalize_package():
    print("--- Phase 10: Publication & Reproducibility ---")
    base_dir = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI"
    results_dir = os.path.join(base_dir, "data", "results")
    
    # 1. Create Structure
    dirs = [
        "tables", "figures", "reproducibility", "manuscript", 
        "reproducibility/models", "reproducibility/data", "configs"
    ]
    for d in dirs:
        os.makedirs(os.path.join(results_dir, d), exist_ok=True)
    
    # 2. Build Tables (Part A)
    print("Part A: Building Tables...")
    eval_dir = os.path.join(base_dir, "data", "processed", "evaluation")
    ablation_dir = os.path.join(base_dir, "data", "processed", "ablation")
    
    # 10.1 Master Results Table
    if os.path.exists(os.path.join(eval_dir, "cv_summary.csv")):
        shutil.copy(os.path.join(eval_dir, "cv_summary.csv"), os.path.join(results_dir, "tables", "results_table.csv"))
    
    # 10.2 Ablation Table
    if os.path.exists(os.path.join(ablation_dir, "ablation_summary.csv")):
        shutil.copy(os.path.join(ablation_dir, "ablation_summary.csv"), os.path.join(results_dir, "tables", "ablation_table.csv"))
        
    # 10.3 Significance Table
    if os.path.exists(os.path.join(eval_dir, "significance_test.csv")):
        shutil.copy(os.path.join(eval_dir, "significance_test.csv"), os.path.join(results_dir, "tables", "significance_table.csv"))

    # 3. Figures (Part B)
    print("Part B: Consolidating Figures...")
    # Move already generated maps to figures folder
    map_dir = os.path.join(results_dir, "maps")
    fig_dir = os.path.join(results_dir, "figures")
    
    for f in ["comparison_plot.png", "gnn_correction_map.png"]:
        src = os.path.join(results_dir, f)
        if os.path.exists(src):
            shutil.move(src, os.path.join(fig_dir, f))
            
    # 4. Reproducibility (Part C)
    print("Part C: Creating Reproducibility Package...")
    repro_dir = os.path.join(results_dir, "reproducibility")
    
    # 10.12 Environment
    with open(os.path.join(repro_dir, "requirements.txt"), 'w') as f:
        f.write("torch\ntorch-geometric\nscikit-learn\npandas\nnumpy\nrasterio\nshap\nesda\nlibpysal\nmatplotlib\nscipy")
        
    # 10.13 Models
    model_src = os.path.join(base_dir, "data", "processed", "models", "pegnn")
    if os.path.exists(model_src):
        for m in os.listdir(model_src):
            shutil.copy(os.path.join(model_src, m), os.path.join(repro_dir, "models", m))
            
    # 10.14 Data
    data_files = [
        "processed/rothc/gnn_dataset.csv",
        "processed/graph/graph_data.pt"
    ]
    for df in data_files:
        src = os.path.join(base_dir, "data", df)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(repro_dir, "data", os.path.basename(df)))
            
    # 10.15 Training Config
    config = {
        "hidden_dim": 128,
        "lr": 0.001,
        "dropout": 0.3,
        "k_neighbors": 8,
        "folds": 5,
        "lambda_phys": 0.1
    }
    with open(os.path.join(results_dir, "configs", "hyperparameters.json"), 'w') as f:
        json.dump(config, f, indent=4)

    # 5. Manuscript (Part D)
    print("Part D: Writing Manuscript Sections...")
    novelty = """
## Novelty Claims for AlphaGraph-Phys
1. Physics-guided residual graph learning: First implementation combining RothC mechanics with GNN residual correction.
2. Positional encoding for spatial SOC: Explicitly encoding geography into the message-passing framework.
3. Adaptive spatial dependency modeling: Multi-scale graph aggregation for cross-landscape SOC variance.
4. Uncertainty-aware SOC mapping: Calibrated MC-Dropout for physically consistent confidence bands.
5. Leakage-resistant spatial validation: Rigorous KMeans-based spatial blocking for soil mapping.
"""
    with open(os.path.join(results_dir, "manuscript", "novelty_claims.md"), 'w') as f:
        f.write(novelty)

    print("Final Package Ready in data/results/")

if __name__ == "__main__":
    finalize_package()
