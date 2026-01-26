# Coral Cover Economics

A Python package for analyzing the economic impacts of coral reef degradation under climate change scenarios. This project combines hierarchical Bayesian beta regression modeling of coral cover (from Sully et al.) with economic valuation to project tourism revenue losses as reefs decline.

## What It Does

This toolkit helps you:
- **Model coral cover** using hierarchical Bayesian beta regression based on environmental variables
- **Project future coral cover** under RCP 4.5 and 8.5 climate scenarios
- **Calculate economic losses** from coral decline using multiple depreciation models (linear, compound, tipping point)
- **Track cumulative impacts** over time, distinguishing between year-over-year losses and opportunity costs
- **Visualize results** with static plots and an interactive web dashboard

## Project Structure

```
.
├── data/
│   └── sully_2022/          # Coral cover and economic data
├── src/
│   ├── models/              # Bayesian beta regression model
│   ├── economics/           # Economic impact analysis
│   │   ├── depreciation_models.py    # Value loss models
│   │   ├── cumulative_impact.py      # Time-integrated losses
│   │   ├── plotting.py                # Static visualizations
│   │   └── run_economic_analysis.py   # Main analysis pipeline
│   ├── plots/               # Plotting utilities and configs
│   └── processing/          # Data loading and preprocessing
├── notebooks/               # Jupyter notebooks for analysis
├── docs/
│   └── index.html          # Interactive web dashboard
└── results/                # Generated figures and data
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the Analysis

```python
from src.economics.run_economic_analysis import run_full_analysis

# Run complete pipeline: load data, calculate losses, generate plots
run_full_analysis()
```

This will:
1. Load coral cover projections and tourism value data
2. Calculate economic losses under different scenarios and models
3. Generate cumulative impact trajectories
4. Create static plots and export data for the web dashboard
5. Save everything to `results/run_{timestamp}/`

### View Interactive Dashboard

Open `docs/index.html` in a browser to explore:
- Cumulative and annual loss trajectories
- Country-level impacts and GDP comparisons
- Interactive spatial maps of site-level losses
- Model comparisons across scenarios

## Key Features

### Depreciation Models

Three models for how economic value declines with coral cover:
- **Linear**: Constant rate of loss per percent coral cover decline
- **Compound**: Accelerating losses as cover decreases
- **Tipping Point**: Catastrophic collapse after crossing a threshold

### Cumulative Impact Analysis

Tracks economic losses over time, distinguishing:
- **Annual value lost**: Year-over-year decline in revenue
- **Opportunity cost**: Baseline revenue lost once a reef collapses
- **Cumulative loss**: Total economic impact over the projection period

### Visualizations

**Static plots** (Matplotlib):
- Scenario comparisons by model and GDP impact
- Country-level loss rankings
- Spatial distributions with customizable colormaps
- Trajectory plots showing value over time

**Interactive dashboard** (HTML/JavaScript):
- Dynamic filtering by scenario, model, and metric
- Responsive maps with viewport-based rendering
- Real-time updates when changing parameters

## Configuration

Edit `CONFIG` in `src/economics/run_economic_analysis.py` to customize:
- Climate scenarios (RCP 4.5/8.5, years 2050/2100)
- Depreciation model parameters
- Output directories and formats
- Country aggregation settings

## Dependencies

- Python >= 3.8
- NumPy, Pandas, GeoPandas
- Matplotlib, Plotly
- PyMC (Bayesian modeling)
- Cartopy (spatial plotting)
- Leaflet.js (interactive maps)

See `requirements.txt` for full list.

## Citation

If you use this code, please cite:

Sully et al. - "Present and future bright and dark spots for coral reefs through climate change"

## Notes

**Important**: This codebase imports `reefshift.utils.config` which is not included in this repository. You may need to:
1. Install an external `reefshift` package if available, or
2. Create a local `reefshift/utils/config.py` module with appropriate path configurations

See `src/models/hb_beta_model.py` for the import statement.
