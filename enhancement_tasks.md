# PINN vs LINDA Comparison Enhancement Tasks

This document outlines tasks to improve the comparison between Physics-Informed Neural Networks (PINN) and traditional LINDA for precipitation nowcasting.

---

## Table of Contents

1. [Dataset Expansion](#1-dataset-expansion)
2. [Architecture Variations](#2-architecture-variations)
3. [Baseline Models](#3-baseline-models)
4. [Hyperparameter Optimization](#4-hyperparameter-optimization)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Integration Guide](#6-integration-guide)

---

## 1. Dataset Expansion

### 1.1 Add Multiple Radar Datasets

**What to do:**
- Add support for multiple pysteps radar datasets (MRMS, Danish, German)
- Create dataset loader functions for each source
- Enable dataset selection via UI

**Why:**
- Different climate regimes test generalization
- Various spatial/temporal resolutions
- Different precipitation types (convective vs stratiform)

**How to integrate:**

```python
# Add to app.py

def load_dataset(dataset_name="swiss"):
    """Load different radar datasets from pysteps"""
    datasets_config = {
        "swiss": {"root": "mch", "date": "201609080000"},
        "mrms": {"root": "mrms", "date": "201906100000"},
        "danish": {"root": "danish", "date": "201606010000"},
        "german": {"root": "german", "date": "201608120000"},
    }
    
    config = datasets_config[dataset_name]
    root_path = pysteps.datasets.download_pysteps_data()
    fns = pysteps.datasets.create_file_list(
        root_path, config["root"], config["date"], 
        timestep=5
    )
    importer = io.get_method(config["root"])
    rainrate_sequence, _, metadata = io.read_timeseries(fns, importer)
    return rainrate_sequence, metadata
```

**Files to modify:** `app.py` (lines 579-640)

---

### 1.2 Event Type Classification

**What to do:**
- Classify events as convective vs stratiform
- Filter datasets by precipitation intensity
- Create balanced test sets

**Why:**
- Convective events: high intensity, localized
- Stratiform events: low intensity, widespread
- Models may perform differently on each type

**How to integrate:**

```python
def classify_event_type(rainrate_sequence, metadata):
    """Classify precipitation event type"""
    max_intensity = np.max(rainrate_sequence)
    spatial_extent = np.mean(rainrate_sequence > 1.0)
    
    if max_intensity > 20 and spatial_extent < 0.1:
        return "convective"
    elif max_intensity < 10 and spatial_extent > 0.3:
        return "stratiform"
    else:
        return "mixed"
```

---

### 1.3 Multi-Resolution Testing

**What to do:**
- Test at different spatial resolutions (0.5km, 1km, 2km)
- Downsample/upsample data systematically
- Evaluate scale dependence

**Why:**
- Resolution affects physical processes
- Models may have different scale preferences

---

### 1.4 Seasonal/Temporal Split

**What to do:**
- Split data by season (summer vs winter)
- Train on one season, test on another
- Evaluate temporal generalization

**Why:**
- Different precipitation mechanisms by season
- Tests model robustness to distribution shift

---

## 2. Architecture Variations

### 2.1 Standard PINN (Baseline)

**What to do:**
- Implement vanilla PINN without integral term
- Use standard PDE residuals (advection-diffusion)
- Compare against LINDA-PINN

**Why:**
- Baseline for physics-informed approaches
- Isolate effect of integral term

**How to integrate:**

```python
class StandardPINN(nn.Module):
    def __init__(self, layers=[2, 256, 256, 1]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_uniform_(self.layers[i].weight)
    
    def compute_pde_residual(self, R, x, y, t):
        """Compute PDE residual for advection-diffusion"""
        R.grad.requires_grad_(True)
        
        # First derivatives
        dR_dt = torch.autograd.grad(R, t, grad_outputs=torch.ones_like(R))[0]
        dR_dx = torch.autograd.grad(R, x, grad_outputs=torch.ones_like(R))[0]
        dR_dy = torch.autograd.grad(R, y, grad_outputs=torch.ones_like(R))[0]
        
        # Second derivatives
        d2R_dx2 = torch.autograd.grad(dR_dx, x, grad_outputs=torch.ones_like(R))[0]
        d2R_dy2 = torch.autograd.grad(dR_dy, y, grad_outputs=torch.ones_like(R))[0]
        
        # PDE: dR/dt + u*dR/dx + v*dR/dy = D*(d2R/dx2 + d2R/dy2)
        u, v, D = 1.0, 1.0, 0.1
        residual = dR_dt + u*dR_dx + v*dR_dy - D*(d2R_dx2 + d2R_dy2)
        
        return torch.mean(residual**2)
```

---

### 2.2 ConvPINN (Convolutional PINN)

**What to do:**
- Replace MLP with convolutional layers
- Add physics constraints to CNN output
- Preserve spatial structure

**Why:**
- CNNs capture spatial correlations better
- More suitable for grid data
- Better inductive bias for weather data

**How to integrate:**

```python
class ConvPINN(nn.Module):
    def __init__(self, channels=[1, 32, 64, 128, 1]):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.conv_layers.append(
                nn.Conv2D(channels[i], channels[i+1], kernel_size=3, padding=1)
            )
        self.physics_layer = PhysicsConstraint()
    
    def forward(self, R_field, metadata):
        x = R_field.unsqueeze(0).unsqueeze(0)  # Add batch/channel dims
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        return x.squeeze()
```

---

### 2.3 U-Net PINN

**What to do:**
- Implement U-Net architecture with skip connections
- Add physics loss at multiple scales
- Multi-resolution feature extraction

**Why:**
- Captures both local and global features
- Proven effective for weather prediction
- Better gradient flow

---

### 2.4 Fourier Neural Operator (FNO)

**What to do:**
- Implement FNO layers for operator learning
- Learn solution operator for IDE
- Compare against point-wise PINN

**Why:**
- Resolution-invariant predictions
- Efficient for PDE solution operators
- State-of-the-art for PDE learning

---

### 2.5 Ablation Studies

**What to do:**
- LINDA-PINN without physics loss
- LINDA-PINN without integral term
- LINDA-PINN without advection

**Why:**
- Quantify contribution of each component
- Understand what makes PINN effective

---

## 3. Baseline Models

### 3.1 ConvLSTM

**What to do:**
- Implement ConvLSTM for sequence prediction
- Train on same data as PINN
- Compare performance

**Why:**
- Standard deep learning baseline
- Widely used in weather prediction

---

### 3.2 TrajGRU

**What to do:**
- Implement trajectory-based GRU
- Use motion field for warping
- Compare against LINDA

**Why:**
- State-of-the-art for precipitation nowcasting
- Directly comparable to LINDA

---

### 3.3 Simple Persistence

**What to do:**
- Predict future = last observed frame
- Compute baseline metrics

**Why:**
- Simple baseline for short-term prediction
- Easy to beat for longer horizons

---

## 4. Hyperparameter Optimization

### 4.1 PINN Hyperparameter Search

**What to do:**
- Search over: learning rate, layers, hidden size, physics weight
- Use Bayesian optimization or grid search
- Track best configurations

**Why:**
- PINN performance sensitive to hyperparameters
- Find optimal balance between data and physics loss

**How to integrate:**

```python
from optuna import create_study

def optimize_pinn_hyperparams(rainrate_sequence, metadata):
    """Hyperparameter optimization for PINN"""
    
    def objective(trial):
        # Sample hyperparameters
        lr = trial.log_uniform("lr", 1e-5, 1e-2)
        hidden_size = trial.choice("hidden_size", [64, 128, 256, 512])
        num_layers = trial.int("num_layers", 3, 8)
        physics_weight = trial.log_uniform("physics_weight", 0.01, 1.0)
        weight_decay = trial.log_uniform("weight_decay", 1e-6, 1e-3)
        
        # Train model
        trainer = LINDAPINNTrainer()
        trainer.model = LINDAPINNModel(layers=[4] + [hidden_size]*num_layers + [1])
        trainer.optimizer = torch.optim.Adam(
            trainer.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        losses, _ = trainer.train_on_radar_sequence(
            rainrate_sequence, metadata, epochs=20, verbose=False
        )
        
        # Return validation loss
        return np.mean(losses[-5:])  # Average of last 5 epochs
    
    study = create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    
    return study.best_params
```

**Files to modify:** `app.py` (add new module or functions)

**Dependencies:** `pip install optuna`

---

### 4.2 LINDA Hyperparameter Search

**What to do:**
- Search over: ensemble members, perturbation parameters
- Optimize for specific metrics (RMSE, CRPS)
- Compare optimized LINDA vs PINN

**Why:**
- Fair comparison requires optimized baselines
- LINDA parameters affect uncertainty quantification

---

### 4.3 Joint Optimization

**What to do:**
- Optimize both models simultaneously
- Use same train/test split
- Compare best configurations

**Why:**
- Fair comparison under equal tuning effort

---

### 4.4 Hyperparameter UI

**What to do:**
- Add Optuna visualization to Gradio app
- Show parameter importance
- Enable re-running optimization

**How to integrate:**

```python
def run_hyperparameter_search(use_pinn=True, n_trials=50):
    """Run hyperparameter optimization with UI"""
    
    if use_synthetic_data:
        rainrate_sequence, metadata = generate_synthetic_data()
    else:
        rainrate_sequence, metadata = load_swiss_radar_data()
    
    if use_pinn:
        best_params = optimize_pinn_hyperparams(rainrate_sequence, metadata)
    else:
        best_params = optimize_linda_hyperparams(rainrate_sequence, metadata)
    
    # Create visualization
    fig = plot_optuna_results(study)
    
    return best_params, fig
```

---

## 5. Evaluation Metrics

### 5.1 Add Comprehensive Metrics

**What to do:**
- Add CRPS (Continuous Ranked Probability Score)
- Add Brier Score for threshold events
- Add spatial metrics (FSS, SSIM)
- Add temporal metrics (autocorrelation)

**Why:**
- Different metrics capture different aspects
- Ensemble metrics important for LINDA
- Spatial metrics capture structure quality

**How to integrate:**

```python
def compute_crps(predictions, ground_truth):
    """Compute Continuous Ranked Probability Score"""
    # For ensemble predictions
    if predictions.ndim == 4:  # (ensemble, time, ny, nx)
        crps = 0
        for t in range(predictions.shape[1]):
            pred_sorted = np.sort(predictions[:, t], axis=0)
            n = pred_sorted.shape[0]
            ranks = np.arange(1, n+1)
            gt = ground_truth[t]
            
            crps_t = np.mean(
                (2 * ranks - n - 1) * pred_sorted - 
                n * np.abs(pred_sorted - gt)
            )
            crps += crps_t
        return crps / predictions.shape[1]
    else:
        return np.mean((predictions - ground_truth)**2)
```

---

### 5.2 Skill Scores

**What to do:**
- Compute skill scores relative to persistence
- Normalize by baseline performance
- Enable cross-dataset comparison

---

### 5.3 Uncertainty Quantification

**What to do:**
- Compute prediction intervals
- Evaluate calibration
- Compare PINN uncertainty vs LINDA ensemble

---

## 6. Integration Guide

### 6.1 Code Organization

**Recommended structure:**

```
pinn_di_dt4lc/
├── app.py                    # Main Gradio app
├── models/
│   ├── __init__.py
│   ├── linda_pinn.py         # Current LINDA-PINN
│   ├── standard_pinn.py      # Standard PINN
│   ├── conv_pinn.py          # Convolutional PINN
│   ├── convlstm.py           # ConvLSTM baseline
│   └── trajgru.py            # TrajGRU baseline
├── data/
│   ├── __init__.py
│   ├── loaders.py            # Dataset loaders
│   ├── preprocessing.py      # Data preprocessing
│   └── event_classification.py # Event type classification
├── training/
│   ├── __init__.py
│   ├── trainers.py           # Training loops
│   ├── hyperparam_search.py  # HPO utilities
│   └── losses.py             # Loss functions
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py            # Evaluation metrics
│   └── visualization.py      # Plotting utilities
└── configs/
    ├── default.yaml
    ├── hpo.yaml
    └── datasets.yaml
```

---

### 6.2 Step-by-Integration

**Phase 1: Dataset Expansion (Priority: High)**
1. Add `load_dataset()` function with multiple dataset support
2. Add dataset selector to Gradio UI
3. Test each dataset loads correctly
4. Document dataset characteristics

**Phase 2: Baseline Models (Priority: High)**
1. Implement ConvLSTM as first baseline
2. Add persistence baseline (trivial)
3. Integrate into comparison UI
4. Run initial comparisons

**Phase 3: Architecture Variations (Priority: Medium)**
1. Implement Standard PINN (ablation)
2. Implement ConvPINN
3. Compare architectures on same datasets
4. Document findings

**Phase 4: Hyperparameter Optimization (Priority: Medium)**
1. Install Optuna
2. Implement `optimize_pinn_hyperparams()`
3. Add HPO UI tab
4. Run optimization for key datasets
5. Document best configurations

**Phase 5: Comprehensive Evaluation (Priority: Low)**
1. Add CRPS, Brier Score, FSS
2. Add skill scores
3. Update UI to show all metrics
4. Create summary tables

---

### 6.3 Testing Strategy

**For each new component:**
1. Unit test: Verify component works in isolation
2. Integration test: Verify component works with existing code
3. Regression test: Ensure existing functionality unchanged
4. Performance test: Measure training time, memory usage

**Example test:**

```python
def test_load_dataset():
    """Test dataset loader"""
    seq, meta = load_dataset("swiss")
    assert seq.ndim == 3
    assert "xpixelsize" in meta
    assert np.all(np.isfinite(seq))
```

---

### 6.4 Documentation

**For each addition:**
1. Add docstring to functions
2. Update this README with results
3. Document hyperparameter choices
4. Note any failures or limitations

---

## Quick Start Commands

```bash
# Install additional dependencies
pip install optuna xarray zarr

# Run with new dataset
python app.py  # Select dataset in UI

# Run hyperparameter optimization
python -c "from app import optimize_pinn_hyperparams; optimize_pinn_hyperparams(...)"

# Test new architecture
python -c "from models.conv_pinn import ConvPINN; model = ConvPINN(); print(model)"
```

---

## Timeline Estimates

| Task | Estimated Time | Priority |
|------|---------------|----------|
| Dataset expansion | 2-4 hours | High |
| Baseline models (ConvLSTM) | 4-6 hours | High |
| Architecture variations | 8-12 hours | Medium |
| Hyperparameter optimization | 4-6 hours | Medium |
| Comprehensive metrics | 2-4 hours | Low |

---

## References

- pysteps documentation: https://pysteps.github.io/
- PINN paper: Raissi et al., 2019
- FNO paper: Li et al., 2021
- TrajGRU paper: Zhang et al., 2022