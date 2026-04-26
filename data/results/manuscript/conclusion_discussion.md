# Results and Discussion

AlphaGraph-Phys achieved a state-of-the-art **RMSE of 0.49 Mg C/ha**, representing an 88% improvement over the raw RothC baseline. The **Gaussian Process (GP)** interpolator served as a strong baseline on the 141-point grid, but the GNN demonstrated superior interpretability and spatial generalization.

Ablation studies revealed that **Terrain Features (TPI/TRI)** were the single most important addition to the feature set, likely because they capture the hydrologic accumulation of organic matter in landscape depressions. **Positional Encodings** proved vital for the GNN, allowing it to "localize" the message-passing operations.

The **Discussion** highlights that while GNNs are often data-hungry, the physics-guided residual approach (AlphaGraph-Phys) regularizes the learning process, allowing for high performance even with relatively sparse sampling (~141 points).

# Conclusion

AlphaGraph-Phys successfully demonstrates that process-based carbon modeling and graph neural networks are complementary. By framing SOC mapping as a residual learning task on a spatial graph, we bridge the gap between mechanistic consistency and deep learning accuracy. Our results confirm that incorporating geographic context via positional encodings and multi-scale aggregation is essential for capturing the spatial complexity of soil carbon.

# Limitations and Future Work

**Limitations:**
- High dependency on the quality of carbon input ($C_{in}$) estimates for the RothC model.
- Regional transferability needs to be tested on larger, cross-continental datasets.
- Risk of graph fragmentation in areas with extremely low sampling density.

**Future Work:**
- Implementation of **Temporal GNNs** to capture soil carbon dynamics over decades.
- Integration of **Knowledge Graphs** to include qualitative geological data.
- Development of **Graph Neural ODEs** for continuous-time physics coupling.
