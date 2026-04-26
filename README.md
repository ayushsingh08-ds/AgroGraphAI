# AgroGraphAI

I built this to try and map Soil Organic Carbon (SOC) at a 10m resolution without losing the underlying physics that most ML models just ignore. The main idea here is to combine a standard process-based model (RothC) with a Multi-Scale Graph Neural Network (GNN). 

Most people either use pure physics models—which are "correct" but way too smooth and miss local landscape features—or they use pure ML like Random Forests, which can overfit and don't understand things like carbon turnover rates. This project uses a hybrid approach: RothC handles the first-order carbon dynamics, and a GNN predicts the residuals (the errors) by looking at the local landscape graph.

### Why this approach?
I decided to go with residual learning because it's much easier for a neural net to correct a decent baseline than to learn complex soil chemistry from scratch. The GNN is specifically useful here because soil isn't just a grid; it's a connected landscape where water and carbon flow based on topography. By building a spatial graph of the terrain, the model can "see" how a point on a hill might affect the carbon in a valley.

### How it works
1. **The Baseline**: We run a RothC-26.3 simulation to get a steady-state SOC estimate. This takes climate and land-use data as input.
2. **Residual Calculation**: We take the ground truth (from SoilGrids or local samples) and subtract the RothC prediction. That difference is what the GNN tries to solve.
3. **Graph Construction**: Each 10m pixel is a node. Edges are built based on spatial proximity and topographic similarity (TPI, TRI, TWI).
4. **The GNN**: A GraphSAGE-based architecture processes these nodes, pulling in Sentinel-2 spectral data and terrain indices to predict the residual.
5. **Final Output**: RothC Baseline + GNN Residual = High-res SOC Map.

### Tech Stack
* **Language**: Python 3.10+
* **Deep Learning**: PyTorch & PyTorch Geometric (for the GraphSAGE implementation)
* **GIS/Raster**: Rasterio, GDAL, GeoPandas (the usual suspects)
* **Physics Engine**: A custom Python implementation of the RothC-26.3 turnover model
* **Data**: Sentinel-2 (spectral), TerraClimate (weather), and SRTM (topography)

### Repo Structure
```text
AgroGraphAI/
├── data/                  # Raw and processed geospatial layers
├── src/
│   ├── graph_engine/      # Scripts for building the spatial adjacency matrices
│   ├── models/            # RothC pipeline and the GNN architecture
│   ├── preprocessing/     # Feature engineering (Terrain indices, NDVI, etc.)
│   ├── training/          # Training scripts for the GNN and baselines
│   └── visualization/     # Plotting and spatial validation tools
├── notebooks/             # Exploratory analysis and prototyping
├── results/               # Where checkpoints and logs end up
└── manuscript/            # LaTeX source for the related research paper
```

### Getting Started
Setting this up can be a bit of a headache because of the GDAL and PyTorch Geometric dependencies. I'd recommend using a dedicated conda environment.

1. **Install dependencies**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install torch-geometric
   pip install rasterio geopandas scikit-learn
   ```

2. **Run the RothC Spin-up**:
   You need to initialize the soil carbon pools before training the ML part.
   ```bash
   python src/models/run_rothc_pipeline.py --mode spinup
   ```

3. **Train the GNN**:
   Once you have the residuals, run the graph trainer.
   ```bash
   python src/training/train_gnn.py --epochs 100 --lr 0.001
   ```

### Performance & Limitations
In my tests on the Hesaraghatta Grasslands dataset, the hybrid model dropped the RMSE from ~4.2 (RothC alone) down to ~0.49. 

**Trade-offs & Realities:**
* **Graph Size**: Building the graph for huge areas is slow and eats RAM. I've optimized it with multi-scale sampling, but it's still heavy.
* **Data Quality**: If your Sentinel-2 data is cloudy or your DEM is noisy, the GNN will just learn to predict that noise.
* **Experimental**: This is still very much a research project. The training scripts work, but don't expect it to be a one-click production tool yet.

### Future Ideas
* **Multi-temporal Graphs**: Right now the graph is static. I want to see if adding temporal edges (linking the same spot across different months) helps.
* **Transfer Learning**: It would be cool to see if a model trained on one grassland can generalize to a forest without full retraining.

### Contributing
If you're into GNNs or soil science, feel free to open a PR. I'm especially interested in making the graph construction more efficient.

### License
MIT
