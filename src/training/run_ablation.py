import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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

def run_ablation():
    print("--- Phase 9A: Ablation Studies ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base_dir = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI"
    graph_dir = os.path.join(base_dir, "data", "processed", "graph")
    ablation_dir = os.path.join(base_dir, "data", "processed", "ablation")
    os.makedirs(ablation_dir, exist_ok=True)
    
    # Load Main Data
    graph_dict = torch.load(os.path.join(graph_dir, "graph_data.pt"), weights_only=False)
    folds_df = pd.read_csv(os.path.join(graph_dir, "fold_masks.csv"))
    res_df = pd.read_csv(os.path.join(base_dir, "data", "processed", "rothc", "residuals.csv"))
    df_feat = pd.read_csv(os.path.join(base_dir, "data", "processed", "clean_master_engineered.csv"))
    
    # Merge targets into features
    df_all = pd.merge(df_feat, res_df[['site_id', 'soc_observed', 'residual_soc', 'soc_rothc']], 
                      left_index=True, right_index=True)
    df_all.rename(columns={'residual_soc': 'residual_target'}, inplace=True)
    
    fold_ids = folds_df['fold_id'].values
    
    # Feature Groups
    pe_cols = [c for c in df_all.columns if 'pe_' in c or 'norm' in c]
    sat_cols = ['ndvi', 'evi', 'savi', 'bsi', 'ndmi', 'B2', 'B3', 'B4', 'B8', 'B11', 'B12']
    terrain_cols = ['dem', 'slope', 'twi', 'tpi', 'tri', 'aspect']
    rothc_cols = ['soc_rothc']
    
    configs = {
        'Full (AlphaGraph-Phys)': [], # No drop
        'No RothC (Raw SOC Target)': ['soc_rothc'],
        'No PE': pe_cols,
        'No Satellite': sat_cols,
        'No Terrain': terrain_cols,
        'No Weights': [] # Handled in training loop
    }
    
    ablation_results = []
    
    for name, drop_cols in configs.items():
        print(f"Testing Configuration: {name}...")
        
        # Filter features
        current_df = df_all.copy()
        exclude = ['site_id', 'node_id', 'soc_observed', 'residual_target'] + drop_cols
        feature_cols = [c for c in current_df.columns if c not in exclude]
        
        X_raw = current_df[feature_cols].values.astype(np.float32)
        scaler = StandardScaler()
        X_scaled = torch.from_numpy(scaler.fit_transform(X_raw)).to(device)
        
        # Target
        if name == 'No RothC (Raw SOC Target)':
            y = torch.from_numpy(df_all['soc_observed'].values.astype(np.float32)).to(device).unsqueeze(1)
        else:
            y = torch.from_numpy(df_all['residual_target'].values.astype(np.float32)).to(device).unsqueeze(1)
            
        edge_index = graph_dict['edge_index'].to(device)
        
        fold_rmses = []
        for fold in [0, 1]: # Fast evaluation on 2 folds for ablation
            t_mask = (fold_ids != fold)
            v_mask = (fold_ids == fold)
            t_idx = torch.where(torch.from_numpy(t_mask))[0].to(device)
            v_idx = torch.where(torch.from_numpy(v_mask))[0].to(device)
            
            model = GraphSAGEModel(X_scaled.shape[1]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            for epoch in range(200):
                model.train()
                optimizer.zero_grad()
                out = model(X_scaled, edge_index)
                loss = torch.nn.functional.mse_loss(out[t_idx], y[t_idx])
                loss.backward()
                optimizer.step()
                
            model.eval()
            with torch.no_grad():
                pred = model(X_scaled, edge_index)[v_idx].cpu().numpy().flatten()
                y_true = y[v_idx].cpu().numpy().flatten()
                
                # If predicting residual, reconstruct SOC
                if name != 'No RothC (Raw SOC Target)':
                    soc_rothc = df_all['soc_rothc'].values[v_mask]
                    y_pred_soc = soc_rothc + pred
                    y_true_soc = df_all['soc_observed'].values[v_mask]
                else:
                    y_pred_soc = pred
                    y_true_soc = y_true
                
                rmse = np.sqrt(mean_squared_error(y_true_soc, y_pred_soc))
                fold_rmses.append(rmse)
        
        ablation_results.append({
            'configuration': name,
            'rmse': np.mean(fold_rmses),
            'feature_count': len(feature_cols)
        })

    df_results = pd.DataFrame(ablation_results)
    df_results.to_csv(os.path.join(ablation_dir, "ablation_summary.csv"), index=False)
    print("\n--- Ablation Results ---")
    print(df_results)

if __name__ == "__main__":
    run_ablation()
