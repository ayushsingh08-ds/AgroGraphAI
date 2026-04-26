import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np

def visualize_results():
    print("--- Generating Visual Comparison ---")
    
    base_dir = r"C:\Users\AYUSH SINGH\Documents\GitHub\AgroGraphAI"
    map_dir = os.path.join(base_dir, "data", "processed", "results", "maps") # Adjusted if maps are still in data/processed
    # Actually let's assume they were moved to results/maps
    map_dir = os.path.join(base_dir, "results", "maps")
    
    # Save to both results dir and manuscript for LaTeX convenience
    output_path_results = os.path.join(base_dir, "results", "figures", "comparison_plot.png")
    output_path_manuscript = os.path.join(base_dir, "manuscript", "figures", "comparison_plot.png")
    
    maps_config = [
        ("RothC (Physics)", "soc_map_rothc.tif"),
        ("GP (Baseline)", "soc_map_gp.tif"),
        ("Agrograph (Proposed)", "soc_map_graphsage.tif")
    ]
    
    # First pass: Load data and determine global min/max for consistent scaling
    loaded_data = []
    for title, filename in maps_config:
        path = os.path.join(map_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: {filename} not found.")
            loaded_data.append(None)
            continue
            
        with rasterio.open(path) as src:
            data = src.read(1)
            nodata = src.nodata
            data = np.where(data == nodata, np.nan, data)
            loaded_data.append(data)
    
    # Calculate global min/max ignoring NaNs
    all_values = np.concatenate([d.flatten() for d in loaded_data if d is not None])
    v_min = np.nanmin(all_values)
    v_max = np.nanmax(all_values)
    
    print(f"Unified scale range: {v_min:.2f} to {v_max:.2f} Mg C/ha")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (title, _) in enumerate(maps_config):
        data = loaded_data[i]
        if data is None:
            continue
            
        im = axes[i].imshow(data, cmap='YlGn', vmin=v_min, vmax=v_max)
        axes[i].set_title(title, fontsize=14, fontweight='bold')
        axes[i].axis('off')
        
        # Add a colorbar to each for individual reading, but they all share the same scale now
        plt.colorbar(im, ax=axes[i], label='SOC Content (Mg C/ha)')
            
    plt.tight_layout()
    plt.savefig(output_path_results, dpi=300)
    plt.savefig(output_path_manuscript, dpi=300)
    print(f"Comparison plot saved to {output_path_manuscript}")

    # Generate Residual Map (How much the GNN changed the physics)
    print("Generating Residual Map...")
    rothc_path = os.path.join(map_dir, "soc_map_rothc.tif")
    agn_path = os.path.join(map_dir, "soc_map_graphsage.tif")
    
    if os.path.exists(rothc_path) and os.path.exists(agn_path):
        with rasterio.open(rothc_path) as r_src:
            with rasterio.open(agn_path) as g_src:
                r_data = r_src.read(1)
                g_data = g_src.read(1)
                
                # Mask out nodata
                mask = (r_data != r_src.nodata) & (g_data != g_src.nodata)
                residual = np.zeros_like(r_data, dtype=np.float32)
                residual[mask] = g_data[mask] - r_data[mask]
                
                plt.figure(figsize=(6, 5))
                plt.imshow(np.where(mask, residual, np.nan), cmap='RdBu_r')
                plt.colorbar(label='Residual Correction (Mg C/ha)')
                plt.title("GNN Correction to RothC Physics", fontweight='bold')
                plt.axis('off')
                plt.savefig(os.path.join(base_dir, "results", "figures", "gnn_correction_map.png"), dpi=300)
                plt.savefig(os.path.join(base_dir, "manuscript", "figures", "gnn_correction_map.png"), dpi=300)
            
    print("All visualizations complete.")

if __name__ == "__main__":
    visualize_results()
