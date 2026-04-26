# References (Selected)

1. **Poggio, L., et al. (2021).** "SoilGrids 2.0: producing soil property maps with quantified uncertainty, at global scale." *SOIL*.
2. **Coleman, K., & Jenkinson, D. S. (1996).** "RothC-26.3: A model for the turnover of carbon in soil."
3. **Klemmer, K., et al. (2023).** "Positional Encodings for Graph Neural Networks."
4. **Zhang, Y., et al. (2024).** "Physics-guided machine learning for soil organic carbon mapping: A residual learning approach."
5. **Hamilton, W., et al. (2017).** "Inductive Representation Learning on Large Graphs (GraphSAGE)."
6. **Veličković, P., et al. (2018).** "Graph Attention Networks (GAT)."
7. **Zhao, Q., & Efremova, N. (2023).** "Graph Transformers for Digital Soil Mapping."
8. **Mohebbi, A., et al. (2025).** "Spatial Data Leakage in Soil Property Prediction."

# Appendix: Hyperparameters

| Parameter | Value |
| :--- | :--- |
| Model | GraphSAGE |
| Hidden Layers | 2 |
| Hidden Units | 128 |
| Dropout | 0.3 |
| Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| Neighborhood (k) | 8 |
| Epochs | 500 |
| Optimizer | Adam |
| Loss Function | MSE + 0.1 * Phys (Relu) |
