# Introduction

Soil Organic Carbon (SOC) is fundamental to climate regulation as the largest terrestrial carbon pool and is essential for soil fertility and ecosystem services. Accurate mapping of SOC at high spatial resolutions is critical for climate mitigation strategies and sustainable agriculture.

However, SOC mapping is hindered by extreme spatial heterogeneity and the high cost of laboratory measurements, leading to sparse and unevenly distributed datasets.

Traditional Digital Soil Mapping (DSM) has shifted toward machine learning models like Random Forest (RF) and XGBoost. While powerful, these methods treat soil samples as independent observations, failing to capture the intrinsic spatial connectivity and neighborhood effects of soil processes.

Process-based models like RothC offer mechanistic consistency by modeling carbon turnover cycles. Yet, these models often rely on coarse regional data and lack the fine-grained adaptivity required for field-scale mapping.

In this work, we introduce **AlphaGraph-Phys**, a physics-guided residual graph learning pipeline. By predicting the errors of a process-based model using a spatial GNN, we combine mechanistic stability with the high-resolution predictive power of deep learning.

Our core contributions include:
1. A **Residual Graph Learning** framework that bridges process models and GNNs.
2. The use of **Positional Encodings** to handle complex spatial autocorrelation.
3. A **Multi-Scale GraphSAGE** architecture for hierarchical landscape modeling.
4. An **Uncertainty-Aware** mapping approach using calibrated MC-Dropout.

# Methodology

## Data Sources
We integrated multi-source data including:
- **SoilGrids 2.0**: Static soil properties (Clay, Sand, Silt, pH).
- **SRTM (30m)**: Topographic features (Elevation, Slope, Aspect).
- **Sentinel-2**: Vegetation and soil indices (NDVI, EVI, BSI).
- **WorldClim**: Climate variables (Rainfall, Temperature).

## RothC Integration
The RothC-26.3 model was run to equilibrium (500-year spin-up) followed by a forward simulation (2018-2025) to generate a mechanistic SOC baseline ($SOC_{RothC}$). The target for our GNN was the residual: $Residual = SOC_{Observed} - SOC_{RothC}$.

## Graph Construction
We constructed a spatial graph where each soil sample is a node. Edges were defined using a **k-Nearest Neighbors (k=8)** approach, with weights determined by a Gaussian distance kernel: $W_{ij} = \exp(-d_{ij}^2 / 2\sigma^2)$.

## AlphaGraph-Phys Architecture
The model uses a multi-scale GraphSAGE architecture. It incorporates **Sine-Cosine Positional Encodings** of coordinates $(x, y)$ to provide the network with explicit geographic context. The final SOC map is reconstructed by adding the GNN residual prediction back to the RothC baseline.

# Experimental Setup

The experiments were conducted on a workstation equipped with an **NVIDIA GeForce RTX 4050 GPU** and 16GB RAM. The pipeline was implemented in **Python 3.12** using **PyTorch** and **PyTorch Geometric**.

Hyperparameters included a learning rate of 0.001, 30% dropout, and 2 GraphSAGE layers with 128 hidden units. Validation followed a **Blocked Spatial Cross-Validation** (5-fold) using KMeans clustering to prevent spatial data leakage.
