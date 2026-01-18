# Coral Cover Economics

A Python package for hierarchical Bayesian beta regression modeling of coral cover based on environmental variables. This project implements models from Sully et al. ("Present and future bright and dark spots for coral reefs through climate change"), originally written in R/JAGS and now translated to Python using PyMC.

## Project Structure

```
.
├── data/                    # Source code
│   └── sully_2022          # ArcGIS Map Package (.mpkx) file loading
├── src/                    # Source code
│   ├── native_r
│   |   └── 1_run_the_beta_model.Rmd
│   |   └── run_beta_model_from_preprocessed.Rmd
│   ├── __init__.py
│   ├── hb_beta_model.py   # Hierarchical Bayesian Beta Model
│   └── mpkx_loading.py    # ArcGIS Map Package (.mpkx) file loading
├── notebooks/             # Jupyter notebooks for analysis
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata and build configuration
└── README.md              # This file
```

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using poetry (if available)

```bash
poetry install
```

## Usage

The main module provides functionality for:

- **Data Loading**: Load coral cover data from CSV files
- **Data Preprocessing**: Clean and standardize environmental variables
- **Hierarchical Bayesian Beta Regression**: Fit models with hierarchical random effects
- **Future Projections**: Project coral cover under climate scenarios
- **Bright and Dark Spots Analysis**: Identify areas with unexpected coral cover

### Example

```python
from src.hb_beta_model import HierarchicalBetaModel, load_data, clean_data

# Load and prepare data
df = load_data("path/to/data.csv")
df = clean_data(df)

# Fit model
model = HierarchicalBetaModel()
model.fit(X, y, site_idx, region_idx, ...)

# Make predictions
y_pred, y_pred_std = model.predict(X_new, site_idx)
```

## Dependencies

- Python >= 3.8
- NumPy
- Pandas
- GeoPandas
- Matplotlib
- Seaborn
- SciPy
- PyMC (for Bayesian modeling)
- ArviZ (for Bayesian analysis)
- py7zr (for .mpkx file extraction)
- pyogrio (for geodatabase reading)

## Development

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Citation

If you use this code, please cite:

Sully et al. - "Present and future bright and dark spots for coral reefs through climate change"

## License

MIT License

## Interactive Map Visualization

An interactive web-based choropleth map is available in `index.html` for visualizing:
- Reef tourism as percentage of national GDP (logarithmic scale)
- Total reef-associated tourism revenue
- Coral cover projections under RCP scenarios (configurable)

### Setting up the Interactive Map

1. **Export data to JSON** from your Jupyter notebook:
   ```python
   from export_map_data import export_for_map
   
   # Export tourism GDP percentage
   export_for_map(
       corrected_by_country_tourism_value_gdp,
       'data/tourism_gdp_pct.json',
       required_columns=['reef_tourism_gdp_as_pct_of_national_gdp']
   )
   
   # Export total tourism revenue
   export_for_map(
       corrected_by_country_tourism_value_gdp,
       'data/tourism_total_revenue.json',
       required_columns=['approx_price_corrected']
   )
   ```

2. **Host on GitHub Pages**:
   - Go to your repository Settings → Pages
   - Select source branch (usually `main`)
   - Save
   - Access your map at: `https://orlando-code.github.io/coral-cover-economics/`

3. **View locally**: Open `index.html` in a web browser (or use a local server)

See `data/README.md` for detailed data export instructions.

## Notes

**Important**: This codebase imports `reefshift.utils.config` which is not included in this repository. You may need to:
1. Install an external `reefshift` package if available, or
2. Create a local `reefshift/utils/config.py` module with appropriate path configurations

See `src/hb_beta_model.py` line 44 for the import statement.
