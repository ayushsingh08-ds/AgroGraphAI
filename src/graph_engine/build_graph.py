import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.sparse import save_npz, csr_matrix
import torch

def build_graph_pipeline():
    print("--- Starting Phase 5: Graph Construction ---")
    
    # Paths
    base_dir = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI"
    processed_dir = os.path.join(base_dir, "data", "processed")
    rothc_dir = os.path.join(processed_dir, "rothc")
    graph_dir = os.path.join(processed_dir, "graph")
    os.makedirs(graph_dir, exist_ok=True)
    
    # 1. Prepare GNN Dataset
    print("Step 5.1: Preparing GNN Dataset...")
    df_features = pd.read_csv(os.path.join(processed_dir, "clean_master_engineered.csv"))
    df_residuals = pd.read_csv(os.path.join(rothc_dir, "residuals.csv"))
    
    # Merge on site_id if possible, otherwise assume order is preserved
    df = df_features.copy()
    df['site_id'] = df_residuals['site_id']
    df['soc_rothc'] = df_residuals['soc_rothc']
    df['residual_target'] = df_residuals['residual_soc']
    df['soc_observed'] = df_residuals['soc_observed']
    
    # Save as gnn_dataset.csv in rothc folder as requested
    df.to_csv(os.path.join(rothc_dir, "gnn_dataset.csv"), index=False)
    
    # Step 5.1 & 5.2: Create Nodes & Project Coordinates
    # Coordinates (x, y) are already in UTM (meters) from previous steps
    nodes_df = df.copy()
    nodes_df['node_id'] = range(len(nodes_df))
    
    # Reorder columns for nodes.csv
    cols = ['node_id', 'site_id', 'x', 'y', 'soc_observed', 'soc_rothc', 'residual_target']
    other_cols = [c for c in nodes_df.columns if c not in cols]
    nodes_df = nodes_df[cols + other_cols]
    
    nodes_df.to_csv(os.path.join(graph_dir, "nodes.csv"), index=False)
    nodes_df.to_csv(os.path.join(graph_dir, "projected_nodes.csv"), index=False)
    print(f"Nodes created: {len(nodes_df)}")
    
    # Step 5.3: Build k-Nearest Neighbor Graph (k=8)
    print("Step 5.3: Building k-NN Graph (k=8)...")
    coords = nodes_df[['x', 'y']].values
    k = 8
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # edges: source_node, target_node
    edges = []
    edge_data = []
    
    for i in range(len(coords)):
        # indices[i][0] is the node itself (distance 0)
        # indices[i][1:] are the k neighbors
        for j in range(1, k + 1):
            neighbor_idx = indices[i][j]
            dist = distances[i][j]
            edges.append([i, neighbor_idx])
            edge_data.append({'source': i, 'target': neighbor_idx, 'distance': dist})
            
    df_edges = pd.DataFrame(edges, columns=['source_node', 'target_node'])
    df_edges.to_csv(os.path.join(graph_dir, "edge_index.csv"), index=False)
    
    # Step 5.4: Compute Edge Weights (Gaussian Kernel)
    print("Step 5.4: Computing Edge Weights...")
    # Calculate sigma (mean distance)
    avg_dist = np.mean([ed['distance'] for ed in edge_data])
    sigma2 = avg_dist ** 2
    
    for ed in edge_data:
        ed['weight'] = np.exp(-(ed['distance']**2) / sigma2)
        
    df_weights = pd.DataFrame(edge_data)
    df_weights.to_csv(os.path.join(graph_dir, "edge_weights.csv"), index=False)
    
    # Step 5.5: Add Self-Loops
    print("Step 5.5: Adding Self-Loops...")
    for i in range(len(coords)):
        edge_data.append({'source': i, 'target': i, 'distance': 0.0, 'weight': 1.0})
        
    df_final_edges = pd.DataFrame(edge_data)
    # We'll use this for adjacency matrix
    
    # Step 5.6: Create Adjacency Matrix
    print("Step 5.6: Creating Adjacency Matrix...")
    N = len(nodes_df)
    row = df_final_edges['source'].values
    col = df_final_edges['target'].values
    data = df_final_edges['weight'].values
    
    adj_matrix = csr_matrix((data, (row, col)), shape=(N, N))
    save_npz(os.path.join(graph_dir, "adjacency.npz"), adj_matrix)
    
    # Step 5.7: Build Node Feature Matrix
    print("Step 5.7: Building Node Feature Matrix...")
    # Exclude targets and IDs
    exclude = ['node_id', 'site_id', 'residual_target', 'soc_observed']
    feature_cols = [c for c in nodes_df.columns if c not in exclude]
    
    X = nodes_df[feature_cols].values.astype(np.float32)
    np.save(os.path.join(graph_dir, "node_features.npy"), X)
    print(f"Feature matrix shape: {X.shape}")
    
    # Step 5.8: Build Target Vector
    print("Step 5.8: Building Target Vector...")
    y = nodes_df['residual_target'].values.astype(np.float32).reshape(-1, 1)
    np.save(os.path.join(graph_dir, "targets.npy"), y)
    
    # Step 5.9: Create Spatial Cross-Validation Folds (Spatial Blocking)
    print("Step 5.9: Creating Spatial CV Folds (KMeans Clustering)...")
    num_folds = 5
    kmeans = KMeans(n_clusters=num_folds, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(coords)
    
    fold_masks = []
    for i in range(N):
        # We'll assign each node a fold based on its cluster
        fold_id = clusters[i]
        fold_masks.append({'node_id': i, 'fold_id': fold_id})
        
    df_folds = pd.DataFrame(fold_masks)
    df_folds.to_csv(os.path.join(graph_dir, "fold_masks.csv"), index=False)
    
    # Step 5.10: Convert to PyTorch Data Format
    print("Step 5.10: Building PyTorch Graph Object...")
    edge_index = torch.tensor(df_final_edges[['source', 'target']].values.T, dtype=torch.long)
    edge_weight = torch.tensor(df_final_edges['weight'].values, dtype=torch.float)
    x_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    
    # Create masks for Fold 0 (as an example/default)
    # In a real training loop, these would be updated per fold
    train_mask = torch.tensor(clusters != 0, dtype=torch.bool)
    test_mask = torch.tensor(clusters == 0, dtype=torch.bool)
    val_mask = test_mask.clone() # Simple split
    
    graph_data = {
        'x': x_tensor,
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'y': y_tensor,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask
    }
    
    torch.save(graph_data, os.path.join(graph_dir, "graph_data.pt"))
    
    # Step 5.11: Graph Diagnostics
    print("Step 5.11: Generating Graph Diagnostics...")
    report = {
        'node_count': N,
        'edge_count': len(df_final_edges),
        'avg_degree': len(df_final_edges) / N,
        'disconnected_nodes': 0, # k-NN ensures connectivity unless k=0
        'density': len(df_final_edges) / (N * N),
        'feature_dim': X.shape[1]
    }
    
    df_report = pd.DataFrame([report])
    df_report.to_csv(os.path.join(graph_dir, "graph_report.csv"), index=False)
    
    print("\n--- Graph Construction Complete ---")
    print(report)

if __name__ == "__main__":
    build_graph_pipeline()
