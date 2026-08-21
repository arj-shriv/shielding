# Legacy Scripts

These scripts have been archived from the repository root.

| Script | What it did | Why archived | Replacement | Reproducible? |
|--------|-------------|--------------|-------------|---------------|
| `1D_CNN_Model_Training (1).py` | Original upstream training script (Windows paths, BiasedWeights variant) | Non-functional on macOS/Linux; all paths hardcoded to `C:/Users/palchowd/...` | `scripts/train/train_cnn.py` | No — requires Windows HPC environment |
| `Shielding_Optimization_Plot_Generator.py` | Steel→Concrete dose-map contour plot | Non-functional: Windows paths, `plt.show()` only (no file save) | Not yet replaced | No |
| `Shielding_Optimization_Plot_Generator_v2.py` | Concrete→Steel dose-map contour plot (order reversed) | Same as above | Not yet replaced | No |
| `Generation_plot_Flux_Comparison.py` | Multi-material flux comparison subplots (Figure 4/5) | Non-functional: Windows paths, `plt.show()` only | Not yet replaced | No |
