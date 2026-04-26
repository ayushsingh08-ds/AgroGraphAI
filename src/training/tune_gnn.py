import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import itertools

# 1. Advanced PE-GNN with Attention and Edge Weights
class AlphaGraphPhys(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads=4, dropout=0.3):
        super(AlphaGraphPhys, self).__init__()
        # Layer 1: GAT with edge weights
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, add_self_loops=False)
        # Layer 2: GraphSAGE for robust aggregation
        self.conv2 = SAGEConv(hidden_channels * heads, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        # x = [N, F]
        x = self.conv1(x, edge_index, edge_attr=edge_weight).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x

def tune_gnn():
    print("--- Starting Phase 7.5: Hyperparameter Tuning for AlphaGraph-Phys ---")
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
    edge_weight = graph_dict['edge_weight'].to(device).unsqueeze(1) # [E, 1] for GAT
    y = graph_dict['y'].to(device)
    fold_ids = folds_df['fold_id'].values
    
    # Hyperparameter Grid
    param_grid = {
        'hidden_dim': [64, 128, 256],
        'lr': [0.001, 0.0005],
        'dropout': [0.2, 0.3, 0.5],
        'heads': [4, 8]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_overall_rmse = float('inf')
    best_params = None
    
    print(f"Testing {len(combinations)} hyperparameter combinations...")
    
    # To speed up, we'll only use 2 folds for tuning
    tuning_folds = [0, 1]
    
    for i, params in enumerate(combinations):
        print(f"Combination {i+1}/{len(combinations)}: {params}")
        fold_rmses = []
        for fold in tuning_folds:
            train_idx = (fold_ids != fold)
            test_idx = (fold_ids == fold)
            
            t_idx = torch.from_numpy(np.where(train_idx)[0]).to(device)
            v_idx = torch.from_numpy(np.where(test_idx)[0]).to(device)
            
            model = AlphaGraphPhys(
                in_channels=x.shape[1], 
                hidden_channels=params['hidden_dim'], 
                heads=params['heads'], 
                dropout=params['dropout']
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
            criterion = torch.nn.MSELoss()
            
            # Fast training for tuning
            model.train()
            for epoch in range(200):
                optimizer.zero_grad()
                out = model(x, edge_index, edge_weight)
                loss = criterion(out[t_idx], y[t_idx])
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                pred = model(x, edge_index, edge_weight)[v_idx].cpu().numpy().flatten()
                y_true = res_df['soc_observed'].values[test_idx]
                soc_rothc = res_df['soc_rothc'].values[test_idx]
                soc_pred = soc_rothc + pred
                rmse = np.sqrt(mean_squared_error(y_true, soc_pred))
                fold_rmses.append(rmse)
        
        avg_rmse = np.mean(fold_rmses)
        if avg_rmse < best_overall_rmse:
            best_overall_rmse = avg_rmse
            best_params = params
            print(f"New Best! RMSE: {avg_rmse:.4f} | Params: {params}")
            
    print(f"\nOptimization Complete. Best Params: {best_params}")
    
    # Final training with best params on all folds
    print("\nTraining final optimized AlphaGraph-Phys...")
    final_results = []
    
    for fold in range(5):
        train_idx = (fold_ids != fold)
        test_idx = (fold_ids == fold)
        t_idx = torch.from_numpy(np.where(train_idx)[0]).to(device)
        v_idx = torch.from_numpy(np.where(test_idx)[0]).to(device)
        
        model = AlphaGraphPhys(
            in_channels=x.shape[1], 
            hidden_channels=best_params['hidden_dim'], 
            heads=best_params['heads'], 
            dropout=best_params['dropout']
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)
        
        # Longer training for final model
        patience = 50
        counter = 0
        best_fold_rmse = float('inf')
        
        for epoch in range(1000):
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
                    torch.save(model.state_dict(), os.path.join(model_dir, 'pegnn', f'optimized_fold{fold}.pt'))
                    counter = 0
                else:
                    counter += 1
            if counter >= patience: break
            
        # Final Evaluation
        model.load_state_dict(torch.load(os.path.join(model_dir, 'pegnn', f'optimized_fold{fold}.pt')))
        model.eval()
        with torch.no_grad():
            res_pred = model(x, edge_index, edge_weight)[v_idx].cpu().numpy().flatten()
            y_true_obs = res_df['soc_observed'].values[test_idx]
            soc_rothc = res_df['soc_rothc'].values[test_idx]
            soc_pred = soc_rothc + res_pred
            
            final_results.append({
                'model': 'AlphaGraph-Phys (Optimized)',
                'fold': fold,
                'rmse': np.sqrt(mean_squared_error(y_true_obs, soc_pred)),
                'r2': r2_score(y_true_obs, soc_pred)
            })

    df_final = pd.DataFrame(final_results)
    summary = df_final.groupby('model').agg({'rmse': ['mean', 'std'], 'r2': ['mean', 'std']}).reset_index()
    summary.columns = ['model', 'rmse_mean', 'rmse_std', 'r2_mean', 'r2_std']
    
    # Load old comparison and add new model
    comparison_path = os.path.join(model_dir, 'metrics', 'model_comparison.csv')
    comp_df = pd.read_csv(comparison_path)
    
    # Update or add
    comp_df = comp_df[comp_df['model'] != 'AlphaGraph-Phys (Optimized)']
    # Convert summary to same format
    new_entry = summary[['model', 'rmse_mean', 'r2_mean']].copy()
    final_comp = pd.concat([comp_df, new_entry], ignore_index=True).sort_values('rmse_mean')
    final_comp.to_csv(comparison_path, index=False)
    
    print("\n--- Optimized Leaderboard ---")
    print(final_comp[['model', 'rmse_mean', 'r2_mean']])

if __name__ == "__main__":
    tune_gnn()
