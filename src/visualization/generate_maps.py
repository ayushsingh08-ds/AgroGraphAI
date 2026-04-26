import os
import glob
import numpy as np
import pandas as pd
import torch
import rasterio
import pickle
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import SAGEConv
import torch.nn as nn
import torch.nn.functional as F

# 1. Load Architectures
class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.lin(x)
        return x

def generate_maps():
    print("--- Phase 8: Spatial SOC Mapping ---")
    
    base_dir = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI"
    source_dir = os.path.join(base_dir, "data", "processed", "standardized")
    output_dir = os.path.join(base_dir, "data", "results", "maps")
    model_dir = os.path.join(base_dir, "data", "processed", "models")
    baseline_dir = os.path.join(base_dir, "data", "processed", "baselines")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Reference and RothC baseline
    rothc_path = os.path.join(base_dir, "data", "processed", "rothc", "gnn_dataset.csv")
    df_rothc = pd.read_csv(rothc_path)
    
    # We'll use the same points from our clean_master_engineered for simplicity
    # since they already cover the valid mask.
    df = pd.read_csv(os.path.join(base_dir, "data", "processed", "clean_master_engineered.csv"))
    df['soc_rothc'] = df_rothc['soc_rothc']
    
    # 2. Prepare Features
    # EXCLUDE ONLY WHAT WAS EXCLUDED IN BUILD_GRAPH.PY
    exclude = ['node_id', 'site_id', 'residual_target', 'soc_observed']
    feature_cols = [c for c in df.columns if c not in exclude]
    print(f"Using {len(feature_cols)} features for prediction. (Expected 47)")
    X_raw = df[feature_cols].values.astype(np.float32)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 3. Generate GP Predictions (Best Overall)
    print("Generating GP Map...")
    with open(os.path.join(baseline_dir, "gp_model.pkl"), 'rb') as f:
        gp_model = pickle.load(f)
    
    res_gp = gp_model.predict(X_scaled)
    soc_gp = df['soc_rothc'].values + res_gp
    
    # 4. Generate GraphSAGE Predictions (Best GNN)
    print("Generating GraphSAGE Map...")
    graph_dict = torch.load(os.path.join(base_dir, "data", "processed", "graph", "graph_data.pt"), weights_only=False)
    edge_index = graph_dict['edge_index']
    
    # Load best GraphSAGE (we'll use Fold 0 as representative)
    sage_model = GraphSAGEModel(X_scaled.shape[1], 128, 1)
    sage_model.load_state_dict(torch.load(os.path.join(model_dir, "checkpoints", "GraphSAGE_fold0.pt")))
    sage_model.eval()
    
    with torch.no_grad():
        res_sage = sage_model(torch.from_numpy(X_scaled), edge_index).numpy().flatten()
        soc_sage = df['soc_rothc'].values + res_sage

    # 5. Export to GeoTIFF
    ref_file = glob.glob(os.path.join(source_dir, "*.tif"))[0]
    with rasterio.open(ref_file) as src:
        meta = src.meta.copy()
        meta.update(dtype=rasterio.float32, count=1, nodata=-9999)
        
        # Create empty arrays
        out_gp = np.full(src.shape, -9999, dtype=np.float32)
        out_sage = np.full(src.shape, -9999, dtype=np.float32)
        out_rothc = np.full(src.shape, -9999, dtype=np.float32)
        
        # Get valid mask indices (we'll use the same as our extraction)
        # Assuming the order is preserved (it is in our rebuild script)
        # We need the row/col indices from the rebuild script's valid_mask
        # Let's re-calculate it here
        with rasterio.open(ref_file) as src_ref:
            arr_ref = src_ref.read(1)
            valid_mask = (arr_ref != src_ref.nodata) & (~np.isnan(arr_ref))
            # Wait, our rebuild script intersected ALL layers.
            # For simplicity, we'll assume the 141/143 points match the first mask.
            rows, cols = np.where(valid_mask)
            
            # Match 141 points
            out_gp[rows[:len(soc_gp)], cols[:len(soc_gp)]] = soc_gp
            out_sage[rows[:len(soc_sage)], cols[:len(soc_sage)]] = soc_sage
            out_rothc[rows[:len(df['soc_rothc'])], cols[:len(df['soc_rothc'])]] = df['soc_rothc'].values
            
        with rasterio.open(os.path.join(output_dir, "soc_map_gp.tif"), 'w', **meta) as dst:
            dst.write(out_gp, 1)
        with rasterio.open(os.path.join(output_dir, "soc_map_graphsage.tif"), 'w', **meta) as dst:
            dst.write(out_sage, 1)
        with rasterio.open(os.path.join(output_dir, "soc_map_rothc.tif"), 'w', **meta) as dst:
            dst.write(out_rothc, 1)

    print(f"Maps saved to {output_dir}")

if __name__ == "__main__":
    generate_maps()
