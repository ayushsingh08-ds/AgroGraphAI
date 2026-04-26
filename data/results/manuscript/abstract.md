# Abstract: AlphaGraph-Phys

**Problem:** Soil Organic Carbon (SOC) is a critical component of global climate regulation and food security. However, mapping SOC at high spatial resolutions remains challenging due to the complex, non-linear interactions between soil properties, terrain, and climate across varied landscapes.

**Gap:** Current digital soil mapping (DSM) approaches typically rely on black-box machine learning models (e.g., Random Forest, XGBoost) that ignore the spatial connectivity of soil observations and often lack physical consistency with established soil carbon turnover processes. Conversely, process-based models like RothC provide mechanistic consistency but often lack the local adaptivity needed for high-resolution spatial prediction.

**Method:** We propose **AlphaGraph-Phys**, a novel hybrid framework that integrates the mechanistic **RothC-26.3** carbon turnover model with a **Spatial Graph Neural Network (GNN)** using a residual learning approach. Instead of predicting raw SOC, the GNN learns to correct the physics-based residuals. We augment the GNN with **Sine-Cosine Positional Encodings** to handle spatial dependency and utilize a **Multi-Scale GraphSAGE** architecture to capture multi-resolution landscape patterns.

**Results:** Evaluated on a diverse 141-point dataset with 49 multi-source features (Sentinel-2, SRTM, SoilGrids), AlphaGraph-Phys achieved an **RMSE of 0.49 Mg C/ha** and an **R² of 0.96**, significantly outperforming the RothC baseline (RMSE 4.23). Ablation studies confirm that topological features (TPI/TRI) and positional encodings are critical drivers of performance, while MC-Dropout provides calibrated 86.5% uncertainty coverage.

**Impact:** AlphaGraph-Phys delivers physically consistent and spatially robust SOC maps, providing a powerful new tool for precise carbon accounting, sustainable land management, and regional climate mitigation strategies.
