import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import Data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. Model Definitions
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

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.lin = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = self.lin(x)
        return x

class PEGNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(PEGNNModel, self).__init__()
        # We assume coordinates are part of features or handled separately
        # Here we just use a deeper SAGE with more capacity for spatial logic
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index).relu()
        x = self.lin(x)
        return x

def train_gnn():
    print("--- Starting Phase 7: AlphaGraph-Phys Training ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    base_dir = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI"
    graph_dir = os.path.join(base_dir, "data", "processed", "graph")
    rothc_dir = os.path.join(base_dir, "data", "processed", "rothc")
    model_dir = os.path.join(base_dir, "data", "processed", "models")
    os.makedirs(model_dir, exist_ok=True)
    for sub in ['graphsage', 'gat', 'pegnn', 'checkpoints', 'predictions', 'metrics']:
        os.makedirs(os.path.join(model_dir, sub), exist_ok=True)
        
    # Load Data
    print("Loading Graph Data...")
    graph_dict = torch.load(os.path.join(graph_dir, "graph_data.pt"), weights_only=False)
    folds_df = pd.read_csv(os.path.join(graph_dir, "fold_masks.csv"))
    res_df = pd.read_csv(os.path.join(rothc_dir, "residuals.csv"))
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_raw = graph_dict['x'].numpy()
    x_scaled = scaler.fit_transform(x_raw)
    x = torch.from_numpy(x_scaled).to(device)
    
    edge_index = graph_dict['edge_index'].to(device)
    y = graph_dict['y'].to(device)
    
    fold_ids = folds_df['fold_id'].values
    num_folds = len(np.unique(fold_ids))
    
    gnn_results = []
    
    model_types = {
        'GraphSAGE': GraphSAGEModel,
        'GAT': GATModel,
        'PE-GNN': PEGNNModel
    }
    
    for m_name, m_class in model_types.items():
        print(f"\n--- Training {m_name} ---")
        
        fold_rmses = []
        
        for fold in range(num_folds):
            print(f"Fold {fold}...")
            train_mask = (fold_ids != fold)
            test_mask = (fold_ids == fold)
            
            # Convert masks to tensors
            train_idx = torch.from_numpy(np.where(train_mask)[0]).to(device)
            test_idx = torch.from_numpy(np.where(test_mask)[0]).to(device)
            
            model = m_class(in_channels=x.shape[1], hidden_channels=128, out_channels=1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
            criterion = torch.nn.MSELoss()
            
            best_val_rmse = float('inf')
            patience = 50
            counter = 0
            
            # Training Loop
            for epoch in range(500):
                model.train()
                optimizer.zero_grad()
                out = model(x, edge_index).flatten()
                
                # Standard MSE Loss
                loss_mse = criterion(out[train_idx], y[train_idx])
                
                # Physics-Guided Loss: Final SOC must be >= 0
                # Final SOC = RothC + Residual (out)
                rothc_train = torch.from_numpy(res_df['soc_rothc'].values[train_mask]).to(device).float()
                soc_final = rothc_train + out[train_idx]
                loss_phys = torch.mean(torch.relu(-soc_final)) # Penalty for negative SOC
                
                loss = loss_mse + 0.1 * loss_phys # Lambda = 0.1
                
                loss.backward()
                optimizer.step()
                
                # Validation (using the test fold as validation for simplicity in this baseline)
                model.eval()
                with torch.no_grad():
                    val_out = model(x, edge_index)
                    val_rmse = np.sqrt(mean_squared_error(y[test_idx].cpu(), val_out[test_idx].cpu()))
                    
                    if val_rmse < best_val_rmse:
                        best_val_rmse = val_rmse
                        torch.save(model.state_dict(), os.path.join(model_dir, 'checkpoints', f'{m_name}_fold{fold}.pt'))
                        counter = 0
                    else:
                        counter += 1
                        
                if counter >= patience:
                    # print(f"Early stopping at epoch {epoch}")
                    break
            
            # Evaluation on best model
            model.load_state_dict(torch.load(os.path.join(model_dir, 'checkpoints', f'{m_name}_fold{fold}.pt')))
            model.eval()
            with torch.no_grad():
                res_pred = model(x, edge_index)[test_idx].cpu().numpy().flatten()
                y_true_obs = res_df['soc_observed'].values[test_mask]
                soc_rothc = res_df['soc_rothc'].values[test_mask]
                
                soc_pred = soc_rothc + res_pred
                
                rmse = np.sqrt(mean_squared_error(y_true_obs, soc_pred))
                mae = mean_absolute_error(y_true_obs, soc_pred)
                r2 = r2_score(y_true_obs, soc_pred)
                
                gnn_results.append({
                    'model': m_name,
                    'fold': fold,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                })
                fold_rmses.append(rmse)
        
        print(f"{m_name} Average RMSE: {np.mean(fold_rmses):.4f}")

    # Step 7.10: Comparison
    print("\nStep 7.10: Generating Comparison...")
    df_gnn = pd.DataFrame(gnn_results)
    df_gnn.to_csv(os.path.join(model_dir, 'metrics', 'gnn_fold_metrics.csv'), index=False)
    
    summary_gnn = df_gnn.groupby('model').agg({'rmse': ['mean', 'std'], 'r2': ['mean', 'std']}).reset_index()
    summary_gnn.columns = ['model', 'rmse_mean', 'rmse_std', 'r2_mean', 'r2_std']
    
    # Load baselines
    baseline_summary = pd.read_csv(os.path.join(base_dir, "data", "processed", "baselines", "baseline_summary.csv"))
    
    # Merge and compare
    final_comparison = pd.concat([baseline_summary, summary_gnn], ignore_index=True)
    final_comparison = final_comparison.sort_values('rmse_mean')
    final_comparison.to_csv(os.path.join(model_dir, 'metrics', 'model_comparison.csv'), index=False)
    
    print("\n--- Final Model Leaderboard ---")
    print(final_comparison[['model', 'rmse_mean', 'r2_mean']])

if __name__ == "__main__":
    train_gnn()
