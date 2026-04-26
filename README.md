```markdown
# Agrograph: Residual Graph Learning for SOC Mapping

Agrograph is a geospatial framework designed for 10m resolution Soil Organic Carbon (SOC) mapping. It combines the **RothC-26.3** mechanistic model with **Multi-Scale Graph Neural Networks (GNNs)** to capture both biological turnover and landscape-level spatial dependencies.

## Overview

Most SOC mapping approaches rely either purely on physical equations (which can miss local variance) or purely on machine learning (which can ignore physical laws). Agrograph uses a residual learning approach:
1. **Physical Baseline**: The RothC model calculates expected SOC based on climate and soil properties.
2. **Spatial Correction**: A GraphSAGE-based GNN predicts the difference (residual) between the physical model and reality by analyzing topographic connectivity and spectral data.

## Project Structure

* **data/**: Contains raw Sentinel-2, SoilGrids data, and processed feature matrices.
* **src/simulation/**: Code for the RothC-26.3 engine and carbon pool spin-up.
* **src/training/**: Training scripts for the GNN and spatial cross-validation utilities.
* **src/preprocessing/**: Tools for generating topographic indices (TPI, TRI, TWI) and spectral processing.
* **manuscript/**: LaTeX source files and figures for the research paper.

## Technical Details

### Residual Learning
The model is trained to minimize the error of the physical model:
$$\Delta SOC = SOC_{observed} - SOC_{RothC}$$

### Key Features
* **Graph Construction**: Nodes represent spatial points, with edges weighted by topographic distance to simulate the flow of water and carbon.
* **Geographic Encodings**: Uses Sine-Cosine positional encodings to account for spatial non-stationarity.
* **Validation**: Implements **Blocked Spatial 5-Fold Cross-Validation** to ensure the model generalizes across different geographic areas without data leakage.

## Getting Started

### Installation
Requires Python 3.10+.
```bash
git clone [https://github.com/ayushsingh08-ds/AgroGraphAI.git](https://github.com/ayushsingh08-ds/AgroGraphAI.git)
cd AgroGraphAI
pip install -r requirements.txt
```

### Basic Workflow
1.  **Spin-up RothC**: Initialize steady-state carbon pools.
    ```bash
    python src/simulation/run_rothc.py --mode spinup --site [site_name]
    ```
2.  **Train GNN**: Train the residual model.
    ```bash
    python src/training/train_gnn.py --model agrograph --epochs 500 --spatial_cv
    ```

## Performance

| Model | RMSE (Mg C/ha) | $R^2$ | Moran's I (p) |
| :--- | :---: | :---: | :---: |
| RothC (Baseline) | 4.23 | 0.07 | 0.45 (0.01) |
| Random Forest | 0.65 | 0.97 | 0.12 (0.05) |
| **Agrograph (Hybrid)** | **0.49** | **0.96** | **0.03 (0.14)** |

*A Moran's I value near 0 indicates that the model residuals are spatially random, suggesting the GNN has successfully captured the spatial logic of the landscape.*

## Authors
* **Ayush Singh** (eng23ds0098@dsu.edu.in)
* **Sadgi Jaiswal** (eng23ds0082@dsu.edu.in)

*Dayananda Sagar University, Bangalore, India.*

## License
This project is licensed under the MIT License.
```
