import os
import pandas as pd
import numpy as np
import torch
from scipy.sparse import load_npz

def audit_data():
    print("--- Starting Data Audit ---")
    
    base_dir = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI"
    graph_dir = os.path.join(base_dir, "data", "processed", "graph")
    rothc_dir = os.path.join(base_dir, "data", "processed", "rothc")
    
    # 1. Audit nodes.csv
    print("\n[1/6] Auditing nodes.csv...")
    nodes_df = pd.read_csv(os.path.join(graph_dir, "nodes.csv"))
    print(f"Total nodes: {len(nodes_df)}")
    
    nan_counts = nodes_df.isna().sum().sum()
    if nan_counts > 0:
        print(f"WARNING: Found {nan_counts} NaN values in nodes.csv!")
        # Find which columns have NaNs
        cols_with_nans = nodes_df.columns[nodes_df.isna().any()].tolist()
        print(f"Columns with NaNs: {cols_with_nans}")
    else:
        print("Success: No NaNs found in nodes.csv.")
        
    # 2. Audit Node Features (numpy)
    print("\n[2/6] Auditing node_features.npy...")
    X = np.load(os.path.join(graph_dir, "node_features.npy"))
    print(f"Feature matrix shape: {X.shape}")
    if np.isnan(X).any():
        print("WARNING: NaNs found in feature matrix!")
    if np.isinf(X).any():
        print("WARNING: Infinity found in feature matrix!")
    print(f"Feature range: [{X.min():.4f}, {X.max():.4f}]")
    
    # 3. Audit Targets
    print("\n[3/6] Auditing targets.npy...")
    y = np.load(os.path.join(graph_dir, "targets.npy"))
    print(f"Target vector shape: {y.shape}")
    print(f"Target stats: Mean={y.mean():.4f}, Std={y.std():.4f}, Min={y.min():.4f}, Max={y.max():.4f}")
    
    # 4. Audit Graph Topology
    print("\n[4/6] Auditing Graph Topology...")
    edges_df = pd.read_csv(os.path.join(graph_dir, "edge_index.csv"))
    weights_df = pd.read_csv(os.path.join(graph_dir, "edge_weights.csv"))
    
    max_node_idx = len(nodes_df) - 1
    oob_sources = edges_df[edges_df['source_node'] > max_node_idx]
    oob_targets = edges_df[edges_df['target_node'] > max_node_idx]
    
    if len(oob_sources) > 0 or len(oob_targets) > 0:
        print(f"WARNING: Out-of-bounds indices found in edges!")
    else:
        print("Success: All edge indices are within range.")
        
    print(f"Weight range: [{weights_df['weight'].min():.4f}, {weights_df['weight'].max():.4f}]")
    
    # 5. Audit Folds
    print("\n[5/6] Auditing Spatial Folds...")
    folds_df = pd.read_csv(os.path.join(graph_dir, "fold_masks.csv"))
    fold_counts = folds_df['fold_id'].value_counts().sort_index()
    print("Nodes per fold:")
    print(fold_counts.to_string())
    
    # 6. Audit PyTorch Object
    print("\n[6/6] Auditing PyTorch graph_data.pt...")
    data = torch.load(os.path.join(graph_dir, "graph_data.pt"))
    print("Keys in graph object:", data.keys())
    print(f"x shape: {data['x'].shape}")
    print(f"edge_index shape: {data['edge_index'].shape}")
    print(f"y shape: {data['y'].shape}")
    
    # Check consistency
    if data['x'].shape[0] != len(nodes_df):
        print("ERROR: Node count mismatch between tensor and CSV!")
    if data['edge_index'].shape[1] != len(edges_df) + len(nodes_df): # edges + self-loops
        print(f"Edge count: Tensor={data['edge_index'].shape[1]}, CSVs={len(edges_df)} + {len(nodes_df)}")
        
    print("\n--- Audit Complete ---")

if __name__ == "__main__":
    audit_data()
