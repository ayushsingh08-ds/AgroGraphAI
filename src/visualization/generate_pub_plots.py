import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_pub_plots():
    print("--- Generating Publication Figures ---")
    base_dir = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI"
    results_dir = os.path.join(base_dir, "data", "results")
    fig_dir = os.path.join(results_dir, "figures")
    
    # 10.6 RMSE Comparison Bar Chart
    results_path = os.path.join(results_dir, "tables", "results_table.csv")
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        df = df.sort_values('rmse_mean')
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model', y='rmse_mean', data=df, palette='viridis')
        plt.title("Model Performance Comparison (RMSE)", fontsize=14)
        plt.ylabel("RMSE (SOC Residual / Predicted)", fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "rmse_comparison.png"), dpi=300)
        plt.savefig(os.path.join(base_dir, "rmse_comparison.png"), dpi=300)
        print("Generated rmse_comparison.png")

    # 10.7 Feature Importance Plot
    importance_path = os.path.join(base_dir, "data", "processed", "explainability", "feature_ranking.csv")
    if os.path.exists(importance_path):
        df_imp = pd.read_csv(importance_path).head(15)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='combined_score', y='feature', data=df_imp, palette='magma')
        plt.title("Global Feature Importance Ranking", fontsize=14, fontweight='bold')
        plt.xlabel("Combined Importance Score (RF + GNN)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "feature_importance.png"), dpi=300)
        plt.savefig(os.path.join(base_dir, "feature_importance.png"), dpi=300)
        print("Generated feature_importance.png")

if __name__ == "__main__":
    generate_pub_plots()
