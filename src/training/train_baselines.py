import os
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# MLP Definition
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.layers(x)

def train_baselines():
    print("--- Starting Phase 6: Baseline Model Training ---")
    
    # Paths
    base_dir = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI"
    processed_dir = os.path.join(base_dir, "data", "processed")
    rothc_dir = os.path.join(processed_dir, "rothc")
    graph_dir = os.path.join(processed_dir, "graph")
    baseline_dir = os.path.join(processed_dir, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)
    
    # Load Data
    print("Step 6.1 & 6.2: Loading Data and Folds...")
    df = pd.read_csv(os.path.join(rothc_dir, "gnn_dataset.csv"))
    folds_df = pd.read_csv(os.path.join(graph_dir, "fold_masks.csv"))
    
    # Prep X and y
    exclude = ['site_id', 'node_id', 'soc_observed', 'residual_target']
    feature_cols = [c for c in df.columns if c not in exclude]
    X_raw = df[feature_cols].values.astype(np.float32)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    
    y_residual = df['residual_target'].values.astype(np.float32)
    y_observed = df['soc_observed'].values.astype(np.float32)
    soc_rothc = df['soc_rothc'].values.astype(np.float32)
    
    # Fold assignments
    fold_ids = folds_df['fold_id'].values
    num_folds = len(np.unique(fold_ids))
    
    # Storage for results
    results = []
    all_predictions = {
        'RothC': [],
        'RF': [],
        'XGB': [],
        'MLP': [],
        'GP': []
    }
    
    models = {
        'RF': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'XGB': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42),
        'GP': GaussianProcessRegressor(kernel=C(1.0) * RBF(10.0), alpha=0.1, random_state=42)
    }
    
    print(f"Training models on {num_folds} spatial folds...")
    
    for fold in range(num_folds):
        print(f"\nProcessing Fold {fold}...")
        train_idx = (fold_ids != fold)
        test_idx = (fold_ids == fold)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_res, y_test_res = y_residual[train_idx], y_residual[test_idx]
        y_test_obs = y_observed[test_idx]
        rothc_test = soc_rothc[test_idx]
        
        # 1. RothC only
        res_rothc = evaluate_model("RothC", rothc_test, y_test_obs, fold)
        results.append(res_rothc)
        
        # 2. Random Forest
        models['RF'].fit(X_train, y_train_res)
        res_pred = models['RF'].predict(X_test)
        soc_pred = rothc_test + res_pred
        results.append(evaluate_model("RF", soc_pred, y_test_obs, fold))
        
        # 3. XGBoost
        models['XGB'].fit(X_train, y_train_res)
        res_pred = models['XGB'].predict(X_test)
        soc_pred = rothc_test + res_pred
        results.append(evaluate_model("XGB", soc_pred, y_test_obs, fold))
        
        # 4. MLP (PyTorch)
        mlp = SimpleMLP(X.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(mlp.parameters(), lr=0.01)
        
        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train_res).unsqueeze(1)
        
        # Train for 100 epochs
        mlp.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = mlp(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
        mlp.eval()
        with torch.no_grad():
            res_pred = mlp(torch.from_numpy(X_test)).numpy().flatten()
            soc_pred = rothc_test + res_pred
            results.append(evaluate_model("MLP", soc_pred, y_test_obs, fold))
            
        # 5. Gaussian Process
        models['GP'].fit(X_train, y_train_res)
        res_pred = models['GP'].predict(X_test)
        soc_pred = rothc_test + res_pred
        results.append(evaluate_model("GP", soc_pred, y_test_obs, fold))
        
    # Step 6.9: Aggregate Results
    print("\nStep 6.9: Aggregating Results...")
    df_metrics = pd.DataFrame(results)
    df_metrics.to_csv(os.path.join(baseline_dir, "fold_metrics.csv"), index=False)
    
    summary = df_metrics.groupby('model').agg({
        'rmse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'bias': ['mean', 'std']
    }).reset_index()
    
    # Flatten columns
    summary.columns = ['model', 'rmse_mean', 'rmse_std', 'mae_mean', 'mae_std', 'r2_mean', 'r2_std', 'bias_mean', 'bias_std']
    summary.to_csv(os.path.join(baseline_dir, "baseline_summary.csv"), index=False)
    
    print("\n--- Baseline Leaderboard ---")
    print(summary[['model', 'rmse_mean', 'r2_mean']].sort_values('rmse_mean'))
    
    # Step 6.10: Save models (trained on full dataset for final use)
    print("\nStep 6.10: Saving final models...")
    # RF
    models['RF'].fit(X, y_residual)
    with open(os.path.join(baseline_dir, "rf_model.pkl"), 'wb') as f:
        pickle.dump(models['RF'], f)
        
    # XGB
    models['XGB'].fit(X, y_residual)
    with open(os.path.join(baseline_dir, "xgb_model.pkl"), 'wb') as f:
        pickle.dump(models['XGB'], f)
        
    # MLP
    full_mlp = SimpleMLP(X.shape[1])
    optimizer = optim.Adam(full_mlp.parameters(), lr=0.01)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y_residual).unsqueeze(1)
    for _ in range(100):
        optimizer.zero_grad()
        loss = criterion(full_mlp(X_t), y_t)
        loss.backward()
        optimizer.step()
    torch.save(full_mlp.state_dict(), os.path.join(baseline_dir, "mlp_model.pt"))
    
    # GP
    models['GP'].fit(X, y_residual)
    with open(os.path.join(baseline_dir, "gp_model.pkl"), 'wb') as f:
        pickle.dump(models['GP'], f)

def evaluate_model(name, y_pred, y_true, fold):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    return {
        'model': name,
        'fold': fold,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'bias': bias
    }

if __name__ == "__main__":
    train_baselines()
