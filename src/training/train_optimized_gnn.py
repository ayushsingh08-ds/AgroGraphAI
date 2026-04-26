import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

class AlphaGraphPhysFinal(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, heads=8, dropout=0.3):
        super(AlphaGraphPhysFinal, self).__init__()
        # Layer 1: GAT focuses on spatial importance
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Layer 2: Residual GraphSAGE for robust physics learning
        self.conv2 = SAGEConv(hidden_channels * heads, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x1 = self.conv1(x, edge_index, edge_attr=edge_weight).relu()
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.conv2(x1, edge_index).relu()
        x = self.lin1(x2).relu()
        x = self.lin2(x)
        return x

def train_optimized():
    print("--- Training Optimized AlphaGraph-Phys (Final Edition) ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    base_dir = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI"
    graph_dir = os.path.join(base_dir, "data", "processed", "graph")
    rothc_dir = os.path.join(base_dir, "data", "processed", "rothc")
    model_dir = os.path.join(base_dir, "data", "processed", "models")
    
    # Load Data
    graph_dict = torch.load(os.path.join(graph_dir, "graph_data.pt"), weights_only=False)
    folds_df = pd.read_csv(os.path.join(graph_dir, "fold_masks.csv"))
    res_df = pd.read_csv(os.path.join(rothc_dir, "residuals.csv"))
    
    # Scale features
    scaler = StandardScaler()
    x_raw = graph_dict['x'].numpy()
    x_scaled = scaler.fit_transform(x_raw)
    
    x = torch.from_numpy(x_scaled).to(device)
    edge_index = graph_dict['edge_index'].to(device)
    edge_weight = graph_dict['edge_weight'].to(device).unsqueeze(1)
    y = graph_dict['y'].to(device)
    fold_ids = folds_df['fold_id'].values
    
    final_results = []
    
    for fold in range(5):
        print(f"Fold {fold}...")
        train_idx = (fold_ids != fold)
        test_idx = (fold_ids == fold)
        t_idx = torch.from_numpy(np.where(train_idx)[0]).to(device)
        v_idx = torch.from_numpy(np.where(test_idx)[0]).to(device)
        
        model = AlphaGraphPhysFinal(in_channels=x.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        
        best_fold_rmse = float('inf')
        patience = 100
        counter = 0
        
        for epoch in range(2000):
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index, edge_weight)
            loss = F.mse_loss(out[t_idx], y[t_idx])
            loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_out = model(x, edge_index, edge_weight)
                val_rmse = np.sqrt(mean_squared_error(y[v_idx].cpu(), val_out[v_idx].cpu()))
                if val_rmse < best_fold_rmse:
                    best_fold_rmse = val_rmse
                    torch.save(model.state_dict(), os.path.join(model_dir, 'pegnn', f'final_optimized_fold{fold}.pt'))
                    counter = 0
                else:
                    counter += 1
            if counter >= patience: break
            
        # Final Evaluation
        model.load_state_dict(torch.load(os.path.join(model_dir, 'pegnn', f'final_optimized_fold{fold}.pt')))
        model.eval()
        with torch.no_grad():
            res_pred = model(x, edge_index, edge_weight)[v_idx].cpu().numpy().flatten()
            y_true_obs = res_df['soc_observed'].values[test_idx]
            soc_rothc = res_df['soc_rothc'].values[test_idx]
            soc_pred = soc_rothc + res_pred
            
            final_results.append({
                'model': 'AlphaGraph-Phys (Final)',
                'fold': fold,
                'rmse': np.sqrt(mean_squared_error(y_true_obs, soc_pred)),
                'r2': r2_score(y_true_obs, soc_pred)
            })

    df_final = pd.DataFrame(final_results)
    summary = df_final.groupby('model').agg({'rmse': ['mean', 'std'], 'r2': ['mean', 'std']}).reset_index()
    summary.columns = ['model', 'rmse_mean', 'rmse_std', 'r2_mean', 'r2_std']
    
    comparison_path = os.path.join(model_dir, 'metrics', 'model_comparison.csv')
    comp_df = pd.read_csv(comparison_path)
    comp_df = comp_df[comp_df['model'] != 'AlphaGraph-Phys (Final)']
    final_comp = pd.concat([comp_df, summary[['model', 'rmse_mean', 'r2_mean']]], ignore_index=True).sort_values('rmse_mean')
    final_comp.to_csv(comparison_path, index=False)
    
    print("\n--- Optimized Leaderboard ---")
    print(final_comp[['model', 'rmse_mean', 'r2_mean']])

if __name__ == "__main__":
    train_optimized()
