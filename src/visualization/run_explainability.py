import os
import pandas as pd
import numpy as np
import pickle
import shap
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib.pyplot as plt

# GNN Architecture
class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=1):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)

def run_explainability():
    print("--- Phase 9B: Explainability ---")
    base_dir = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI"
    exp_dir = os.path.join(base_dir, "data", "processed", "explainability")
    model_dir = os.path.join(base_dir, "data", "processed", "models")
    baseline_dir = os.path.join(base_dir, "data", "processed", "baselines")
    os.makedirs(exp_dir, exist_ok=True)
    
    # 1. SHAP for Random Forest
    print("Step 9.7: Running SHAP for Random Forest...")
    # Load RF Model
    with open(os.path.join(baseline_dir, "rf_model.pkl"), 'rb') as f:
        rf_model = pickle.load(f)
    
    # Load Data for SHAP (using a subset of 50 samples for speed)
    nodes_df = pd.read_csv(os.path.join(base_dir, "data", "processed", "graph", "nodes.csv"))
    exclude = ['node_id', 'site_id', 'residual_target', 'soc_observed']
    feature_cols = [c for c in nodes_df.columns if c not in exclude]
    X_subset = nodes_df[feature_cols].iloc[:50].values
    
    print(f"Detected {len(feature_cols)} features: {feature_cols[:5]}...")
    
    explainer_rf = shap.TreeExplainer(rf_model)
    shap_values = explainer_rf.shap_values(X_subset, check_additivity=False)
    
    # Save Feature Importance Ranking
    rf_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    rf_importance.to_csv(os.path.join(exp_dir, "feature_ranking_rf.csv"), index=False)
    print("Top RF Features:")
    print(rf_importance.head(10))

    # 2. GNN Explainability (GNNExplainer)
    print("Step 9.8: Running GNNExplainer...")
    graph_dict = torch.load(os.path.join(base_dir, "data", "processed", "graph", "graph_data.pt"), weights_only=False)
    x = graph_dict['x']
    edge_index = graph_dict['edge_index']
    
    # Load Model
    model = GraphSAGEModel(x.shape[1])
    model.load_state_dict(torch.load(os.path.join(model_dir, "checkpoints", "GraphSAGE_fold0.pt")))
    model.eval()
    
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='node',
            return_type='raw',
        ),
    )
    
    # Explain one central node (Node 0)
    explanation = explainer(x, edge_index, index=0)
    
    # Save Feature Importance from GNN
    if hasattr(explanation, 'node_mask') and explanation.node_mask is not None:
        mask = explanation.node_mask.cpu().numpy()
        # If mask is [N, F], take mean across nodes or just the explained node's row
        if mask.ndim == 2:
            mask = mask.mean(axis=0)
        
        if len(mask) == len(feature_cols):
            gnn_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': mask
            }).sort_values('importance', ascending=False)
        else:
            print(f"Warning: Mask length {len(mask)} != feature count {len(feature_cols)}")
            gnn_importance = pd.DataFrame({'feature': feature_cols, 'importance': 0.0})
    else:
        print("Warning: No node_mask found.")
        gnn_importance = pd.DataFrame({'feature': feature_cols, 'importance': 0.0})
    
    gnn_importance.to_csv(os.path.join(exp_dir, "feature_ranking_gnn.csv"), index=False)
    
    # Step 9.9: Generating Final Feature Ranking
    print("Step 9.9: Generating Final Feature Ranking...")
    combined = pd.merge(rf_importance, gnn_importance, on='feature', suffixes=('_rf', '_gnn'))
    combined['combined_score'] = (combined['importance_rf'] + combined['importance_gnn']) / 2
    combined = combined.sort_values('combined_score', ascending=False)
    
    combined.to_csv(os.path.join(exp_dir, "feature_ranking.csv"), index=False)
    print("--- Explainability Complete ---")
    print(combined[['feature', 'combined_score']].head(10))

if __name__ == "__main__":
    run_explainability()
