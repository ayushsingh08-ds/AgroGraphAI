import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance():
    print("--- Generating Publication Quality Feature Importance Plot ---")
    base_dir = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI"
    data_path = os.path.join(base_dir, "data", "processed", "explainability", "feature_ranking.csv")
    output_path = os.path.join(base_dir, "data", "results", "figures", "feature_importance.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return

    df = pd.read_csv(data_path)
    
    # Take top 20 features for clarity
    df_top = df.head(20).copy()
    
    # Rename for cleaner plot
    df_top['feature'] = df_top['feature'].str.replace('_', ' ').str.title()

    plt.figure(figsize=(10, 8), dpi=300)
    sns.set_style("whitegrid")
    
    # Create the horizontal bar chart
    palette = sns.color_palette("viridis", len(df_top))
    ax = sns.barplot(
        x='combined_score', 
        y='feature', 
        data=df_top, 
        palette=palette,
        hue='feature',
        legend=False
    )
    
    plt.title("Global Feature Importance Ranking (Agrograph)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Combined Importance Score (SHAP + GNNExplainer)", fontsize=12)
    plt.ylabel("Environmental Covariate", fontsize=12)
    
    # Add values at the end of bars
    for i, v in enumerate(df_top['combined_score']):
        ax.text(v + 0.01, i + 0.1, f"{v:.3f}", color='black', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_feature_importance()
