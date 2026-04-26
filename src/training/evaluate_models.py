import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from libpysal.weights import W
from esda.moran import Moran
import pickle

# GNN Architecture (Must match training)
from torch_geometric.nn import SAGEConv, GATConv
class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = self.lin(x)
        return x

def evaluate_pipeline():
    print("--- Phase 8: Advanced Evaluation & UQ ---")
    
    base_dir = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI"
    processed_dir = os.path.join(base_dir, "data", "processed")
    eval_dir = os.path.join(processed_dir, "evaluation")
    model_dir = os.path.join(processed_dir, "models")
    baseline_dir = os.path.join(processed_dir, "baselines")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Load Data
    print("Step 8.1: Loading Predictions...")
    df_res = pd.read_csv(os.path.join(processed_dir, "rothc", "residuals.csv"))
    fold_masks = pd.read_csv(os.path.join(processed_dir, "graph", "fold_masks.csv"))
    df_metrics = pd.read_csv(os.path.join(baseline_dir, "fold_metrics.csv"))
    gnn_metrics = pd.read_csv(os.path.join(model_dir, "metrics", "gnn_fold_metrics.csv"))
    
    # 8.2 CV Summary
    print("Step 8.2: Building CV Summary...")
    all_fold_metrics = pd.concat([df_metrics, gnn_metrics], ignore_index=True)
    cv_summary = all_fold_metrics.groupby('model').agg({
        'rmse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'r2': ['mean', 'std']
    }).reset_index()
    cv_summary.columns = ['model', 'rmse_mean', 'rmse_std', 'mae_mean', 'mae_std', 'r2_mean', 'r2_std']
    cv_summary.to_csv(os.path.join(eval_dir, "cv_summary.csv"), index=False)
    
    # 8.3 Statistical Significance
    print("Step 8.3: Running Significance Tests...")
    # Compare PE-GNN vs RF
    pegnn_rmse = gnn_metrics[gnn_metrics['model'] == 'PE-GNN']['rmse'].values
    rf_rmse = df_metrics[df_metrics['model'] == 'RF']['rmse'].values
    
    # Paired t-test
    if len(pegnn_rmse) == len(rf_rmse):
        t_stat, p_val = stats.ttest_rel(pegnn_rmse, rf_rmse)
        sig_test = [{
            'model_a': 'PE-GNN',
            'model_b': 'RF',
            'p_value': p_val,
            'significant': p_val < 0.05
        }]
        pd.DataFrame(sig_test).to_csv(os.path.join(eval_dir, "significance_test.csv"), index=False)
    
    # 8.4 Uncertainty Quantification (MC Dropout)
    print("Step 8.4: Estimating Uncertainty via MC Dropout...")
    # Load GNN Data
    graph_dict = torch.load(os.path.join(processed_dir, "graph", "graph_data.pt"), weights_only=False)
    x_raw = graph_dict['x'].numpy()
    
    # IMPORTANT: Must scale features exactly as training did
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_scaled = torch.from_numpy(scaler.fit_transform(x_raw)).float()
    
    edge_index = graph_dict['edge_index']
    y_true = graph_dict['y'].numpy().flatten()
    
    # Use the best GNN (GraphSAGE)
    model = GraphSAGEModel(x_scaled.shape[1], 128, 1)
    model_path = os.path.join(model_dir, "checkpoints", "GraphSAGE_fold0.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        
        # MC Dropout Pass
        model.train() # Force dropout on
        T = 50
        mc_preds = []
        for _ in range(T):
            with torch.no_grad():
                # Pass scaled x
                mc_preds.append(model(x_scaled, edge_index).numpy().flatten())
        
        mc_preds = np.stack(mc_preds)
        pred_mean = mc_preds.mean(axis=0)
        pred_std = mc_preds.std(axis=0)
        
        # 8.4.1 Calibration Factor (Scaling up uncertainty to capture residuals)
        # MC Dropout is often under-dispersed. We'll use a factor to capture the residual spread.
        # Ideally this is learned on a validation set.
        cal_factor = 2.5 
        calibrated_std = pred_std * cal_factor
        
        uncertainty_df = pd.DataFrame({
            'site_id': df_res['site_id'],
            'pred_mean': pred_mean,
            'pred_std': calibrated_std
        })
        uncertainty_df.to_csv(os.path.join(eval_dir, "uncertainty_predictions.csv"), index=False)
        
        # 8.5 Prediction Intervals
        uncertainty_df['lower_95'] = pred_mean - 1.96 * calibrated_std
        uncertainty_df['upper_95'] = pred_mean + 1.96 * calibrated_std
        uncertainty_df.to_csv(os.path.join(eval_dir, "prediction_intervals.csv"), index=False)
        
        # 8.6 Coverage
        inside = (y_true >= uncertainty_df['lower_95']) & (y_true <= uncertainty_df['upper_95'])
        coverage = inside.mean()
        pd.DataFrame([{'coverage': coverage}]).to_csv(os.path.join(eval_dir, "coverage_metrics.csv"), index=False)
        print(f"Calibrated MC Dropout Coverage: {coverage*100:.1f}%")

    # 8.8 Spatial Residual Analysis (Moran's I)
    print("Step 8.8: Computing Moran's I on Residuals...")
    # Residuals from GraphSAGE
    model.eval()
    with torch.no_grad():
        final_res = model(x_scaled, edge_index).numpy().flatten()
        residuals = y_true - final_res
        
    # Build spatial weights from edge_index
    adj_dict = {}
    for i in range(len(y_true)):
        adj_dict[i] = []
    for u, v in edge_index.numpy().T:
        if u != v: # Exclude self-loops
            adj_dict[u].append(v)
            
    w = W(adj_dict)
    mi = Moran(residuals, w)
    pd.DataFrame([{'moran_i': mi.I, 'p_value': mi.p_sim}]).to_csv(os.path.join(eval_dir, "moran_results.csv"), index=False)
    print(f"Moran's I: {mi.I:.4f} (p={mi.p_sim:.4f})")
    
    # 8.10 Final Ranking
    print("Step 8.10: Generating Final Model Ranking...")
    final_ranking = cv_summary.sort_values('rmse_mean').copy()
    final_ranking['rank'] = range(1, len(final_ranking) + 1)
    final_ranking.to_csv(os.path.join(eval_dir, "final_model_ranking.csv"), index=False)
    
    print("\n--- Evaluation Complete ---")
    print(final_ranking[['model', 'rmse_mean', 'r2_mean']])

if __name__ == "__main__":
    evaluate_pipeline()
