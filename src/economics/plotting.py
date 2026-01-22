"""
Plotting utilities for coral reef economics analysis.

Provides both static (matplotlib) and interactive (plotly) visualizations.
"""

import warnings
from pathlib import Path
from typing import Dict, List

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.patches import Patch

from src import utils
from src.plots import plot_utils

from .analysis import AnalysisResults, DepreciationResult
from .depreciation_models import DepreciationModel

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Color palette (colorblind-friendly)
COLOURS = {
    "rcp45": plot_utils.get_wa_colormap(n_colours=100, index=0),  # Blue
    "rcp85": plot_utils.get_wa_colormap(n_colours=100, index=-1),  # Red
    "tourism": "#2ecc71",  # Green
    "protection": "#9b59b6",  # Purple
    "neutral": "#7f8c8d",  # Gray
}

# Scenario display names
SCENARIO_LABELS = {
    "RCP45_2050": "RCP 4.5 (2050)",
    "RCP45_2100": "RCP 4.5 (2100)",
    "RCP85_2050": "RCP 8.5 (2050)",
    "RCP85_2100": "RCP 8.5 (2100)",
}

# RCP scenario labels (without year)
RCP_LABELS = {
    "RCP45": "RCP 4.5",
    "RCP85": "RCP 8.5",
}

NESTED_SCENARIO_LABELS = {
    "RCP45": {2050: "RCP 4.5 (2050)", 2100: "RCP 4.5 (2100)"},
    "RCP85": {2050: "RCP 8.5 (2050)", 2100: "RCP 8.5 (2100)"},
}

# Scenario comparison plot configuration
SCENARIO_COMPARISON_CONFIG = {
    "model_order": ["Linear", "Compound", "Tipping Point"],
    "year_order": ["2100", "2050"],  # 2100 behind, 2050 overlay
    "rcp_order": ("RCP45", "RCP85"),
    "bar_width": 0.15,
    "country_spacing": 1.4,
    "country_height_factor": 0.7,  # Height per country
    "min_figure_height": 6,
    "model_hatches": {
        "Linear": "",
        "Compound": "///",
        "Tipping Point": "...",
    },
    "bar_alphas": {"2100": 0.35, "2050": 0.85},
    "hatch_linewidth": 0.8,
    "bar_edgecolor": "black",
    "bar_linewidth": 1.2,
    "legend": {
        "fontsize": 14,
        "title_fontsize": 16,
        "scenario_bbox": (0.98, 0.7),
        "model_bbox": (0.912, 0.3),
    },
    "axis": {
        "ylabel_fontsize": 11,
        "xlabel_fontsize": 13,
        "xlabel": "Value Loss (Million USD)",
        "y_tick_pad": 10,
    },
    "grid": {
        "minor_interval": 250,  # Million USD
        "minor_alpha": 0.4,
        "major_alpha": 0.75,
    },
    "vline": {
        "color": "black",
        "linewidth": 1.2,
        "alpha": 0.8,
    },
    "figure": {
        "dpi": 300,
        "save_dpi": 150,
        "bottom_margin": 0.08,
    },
}

# GDP impact comparison plot configuration
GDP_IMPACT_COMPARISON_CONFIG = {
    "figure": {
        "width": 14,
        "min_height": 8,
        "country_height_factor": 0.5,
        "dpi": 300,
        "save_dpi": 150,
        "bottom_margin": 0.08,
    },
    "bar": {
        "total_height": 0.75,
        "edgecolor_no_hatch": "white",
        "edgecolor_hatch": "black",
        "linewidth": 0.5,
    },
    "model_hatches": {
        "Linear": "",
        "Compound": "///",
    },
    "year_alphas": {"2050": 0.6, "2100": 1.0},
    "axis": {
        "ylabel_fontsize": 10,
        "xlabel_fontsize": 12,
        "xlabel": "Projected Value Loss as % of National GDP",
        "title_fontsize": 14,
    },
    "legend": {
        "fontsize": 9,
        "title": "Scenario (Model)",
        "title_fontsize": 10,
        "ncol": 2,
        "loc": "lower right",
    },
    "grid": {
        "alpha": 0.3,
        "linestyle": "--",
    },
    "footer": {
        "text": "Color: RCP scenario (blue=4.5, red=8.5) | Opacity: Year (light=2050, dark=2100) | Hatch: Model (solid=Linear, ///=Compound)",
        "fontsize": 8,
        "x": 0.02,
        "y": 0.02,
        "style": "italic",
        "color": "gray",
    },
}

country_mapping = {
    "N. Mariana Is.": "Northern Mariana Islands",
}
# =============================================================================
# UTILS
# =============================================================================


def parse_scenario(scenario_name, model_name):
    """Extract RCP, year from scenario name."""
    rcp = "RCP45" if "45" in scenario_name else "RCP85"
    year = "2100" if "2100" in scenario_name else "2050"
    if "linear" in model_name.lower():
        model_type = "Linear"
    elif "compound" in model_name.lower():
        model_type = "Compound"
    elif "tipping point" in model_name.lower():
        model_type = "Tipping Point"
    else:
        model_type = "Unknown"

    return rcp, year, model_type


def get_scenario_color(scenario: str) -> str:
    """Get color for a scenario."""
    if "85" in scenario:
        return COLOURS["rcp85"]
    elif "45" in scenario:
        return COLOURS["rcp45"]
    return COLOURS["neutral"]


def format_currency(value: float, precision: int = 1) -> str:
    """Format currency value in human-readable form."""
    if abs(value) >= 1e12:
        return f"${value / 1e12:.{precision}f}T"
    elif abs(value) >= 1e9:
        return f"${value / 1e9:.{precision}f}B"
    elif abs(value) >= 1e6:
        return f"${value / 1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"${value / 1e3:.{precision}f}K"
    return f"${value:.{precision}f}"


def sort_key(item):
    key, result = item
    rcp, year, model = parse_scenario(result.scenario, result.model.name)
    return (rcp, year, model)


# =============================================================================
# MODEL COMPARISON PLOTS
# =============================================================================


def plot_model_comparison(
    models: List[DepreciationModel] = None,
    value: float = 100.0,
    delta_cc_range: np.ndarray = None,
    ax: plt.Axes = None,
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot comparison of depreciation models.

    Shows how different models translate coral cover change to value loss.

    Parameters
    ----------
    models : list, optional
        List of DepreciationModel instances. Default: linear and compound.
    value : float
        Base value for comparison.
    delta_cc_range : array, optional
        Range of coral cover changes to plot. Default: 0 to -100pp.
    ax : Axes, optional
        Matplotlib axes to plot on. Creates new figure if None.
    save_path : Path, optional
        Save figure to this path.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    from .depreciation_models import CompoundModel, LinearModel

    if models is None:
        # models = [LinearModel(), CompoundModel(), TippingPointModel()]
        models = [LinearModel(), CompoundModel()]

    if delta_cc_range is None:
        delta_cc_range = np.linspace(0, -1.0, 101)  # 0 to -100pp

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Convert to percentage points for display
    x_values = delta_cc_range * 100  # Now in percentage points

    for model in models:
        remaining = model.calculate(delta_cc_range, value)
        loss_pct = 100 * (value - remaining) / value
        ax.plot(np.abs(x_values), loss_pct, label=model.name, linewidth=2)

    ax.set_xlabel("Coral Cover Decrease (percentage points)", fontsize=12)
    ax.set_ylabel("Value Loss (%)", fontsize=12)
    ax.set_title("Depreciation Model Comparison", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# STATIC (MATPLOTLIB) PLOTS
# =============================================================================


def plot_country_losses(
    result: DepreciationResult,
    top_n: int = 20,
    ax: plt.Axes = None,
    save_path: Path = None,
) -> plt.Figure:
    """
    Horizontal bar chart of value losses by country.

    Parameters
    ----------
    result : DepreciationResult
        Analysis result to plot.
    top_n : int
        Number of top countries to show.
    ax : Axes, optional
        Matplotlib axes.
    save_path : Path, optional
        Save path.

    Returns
    -------
    Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    else:
        fig = ax.figure

    by_country = result.by_country.head(top_n)
    country_col = (
        "country" if "country" in by_country.columns else by_country.columns[0]
    )

    countries = by_country[country_col].values[::-1]  # Reverse for horizontal bar
    losses = by_country["value_loss"].values[::-1]
    loss_fractions = by_country["loss_fraction"].values[::-1]

    color = get_scenario_color(result.scenario)
    bars = ax.barh(countries, losses / 1e6, color=color, alpha=0.7)

    # Add percentage labels
    for bar, frac in zip(bars, loss_fractions):
        width = bar.get_width()
        ax.text(
            width + 0.01 * ax.get_xlim()[1],
            bar.get_y() + bar.get_height() / 2,
            f"{frac * 100:.1f}%",
            va="center",
            fontsize=9,
            color="gray",
        )

    ax.set_xlabel("Value Loss ($ Million)", fontsize=12)
    ax.set_title(
        f"Top {top_n} Countries by Value Loss\n{result.scenario} | {result.model.name}",
        fontsize=14,
    )
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_scenario_comparison(
    results: AnalysisResults,
    value_type: str = None,
    top_n: int = 15,
    save_path: Path = None,
) -> plt.Figure:
    """
    Compare scenarios side-by-side for top countries with overlaid 2100/2050 bars.

    2050 bars overlay (in darker alpha) up to the corresponding height on 2100.
    Bar widths are wider; models ordered linear, compound, tipping point.

    Parameters
    ----------
    results : AnalysisResults
        Analysis results container.
    value_type : str, optional
        Filter by value type.
    top_n : int
        Number of countries.
    save_path : Path, optional
        Save path.

    Returns
    -------
    Figure
    """
    import matplotlib.colors as mcolors

    cfg = SCENARIO_COMPARISON_CONFIG

    # Extract relevant results
    relevant_results = [
        (k, r)
        for k, r in results.results.items()
        if value_type is None or r.value_type == value_type
    ]

    if len(relevant_results) < 2:
        warnings.warn("Need at least 2 results to compare scenarios")
        return None

    # Collate results into dict {(rcp, model, year): result}
    scenario_dict = {}
    for key, result in relevant_results:
        rcp, year, model_type = parse_scenario(result.scenario, result.model.name)
        scenario_dict[(rcp.upper(), model_type, year)] = result

    # Get top countries from worst-case scenario
    model_order = cfg["model_order"]
    year_order = cfg["year_order"]
    rcp_order = cfg["rcp_order"]

    worst_key = None
    for model_name in model_order:
        worst_key = ("RCP85", model_name, "2100")
        if worst_key in scenario_dict:
            break
    if worst_key not in scenario_dict and len(scenario_dict) > 0:
        worst_key = list(scenario_dict.keys())[0]

    worst_case = scenario_dict[worst_key]
    country_col = (
        "country"
        if "country" in worst_case.by_country.columns
        else worst_case.by_country.columns[0]
    )
    top_countries = worst_case.by_country.head(top_n)[country_col].tolist()

    # Prepare data: {rcp: {model: {year: losses}}}
    data = {rcp: {model: {} for model in model_order} for rcp in rcp_order}
    for rcp in rcp_order:
        for model in model_order:
            for year in year_order:
                res = scenario_dict.get((rcp, model, year))
                if res is not None:
                    byc = res.by_country[
                        res.by_country[country_col].isin(top_countries)
                    ]
                    byc = (
                        byc.set_index(country_col).reindex(top_countries).reset_index()
                    )
                    data[rcp][model][year] = byc["value_loss"].values / 1e6
                else:
                    data[rcp][model][year] = np.zeros(len(top_countries))

    # Setup figure
    fig, ax = plt.subplots(
        figsize=(
            10,
            max(cfg["min_figure_height"], top_n * cfg["country_height_factor"]),
        ),
        dpi=cfg["figure"]["dpi"],
    )

    # Calculate positions and offsets
    bar_width = cfg["bar_width"]
    y_pos = np.arange(len(top_countries)) * cfg["country_spacing"]

    rcp_cluster_gap = 4 * bar_width
    model_gap = bar_width
    rcp_offsets = {
        "RCP45": +rcp_cluster_gap / 2,
        "RCP85": -rcp_cluster_gap / 2,
    }
    model_offsets = {model: (i - 1) * model_gap for i, model in enumerate(model_order)}

    plt.rcParams["hatch.linewidth"] = cfg["hatch_linewidth"]

    # Draw bars
    for model in model_order:
        for rcp in rcp_order:
            y_vals = y_pos + rcp_offsets[rcp] + model_offsets[model]
            color = COLOURS[rcp.lower()]
            hatch = cfg["model_hatches"][model]
            has_hatch = bool(hatch)

            for year, alpha in cfg["bar_alphas"].items():
                losses = data[rcp][model].get(year, np.zeros(len(top_countries)))
                ax.barh(
                    y_vals[::-1],
                    losses,
                    height=bar_width,
                    color=mcolors.to_rgba(color, alpha),
                    hatch=hatch,
                    edgecolor=cfg["bar_edgecolor"] if has_hatch else "none",
                    linewidth=cfg["bar_linewidth"] if has_hatch else 1,
                    zorder=1 if year == "2100" else 3,
                )

    # Draw vertical lines
    cluster_half_height = (
        abs(rcp_offsets["RCP45"] - rcp_offsets["RCP85"]) / 2
        + abs(model_offsets[model_order[-1]])
        + bar_width / 2
    )
    vline_cfg = cfg["vline"]
    for y in y_pos[::-1]:
        ax.vlines(
            x=0,
            ymin=y - cluster_half_height,
            ymax=y + cluster_half_height,
            color=vline_cfg["color"],
            linewidth=vline_cfg["linewidth"],
            alpha=vline_cfg["alpha"],
            zorder=4,
        )

    # Create legends
    legend_cfg = cfg["legend"]

    # Scenario legend (RCP colors + year indicators)
    scenario_handles = [
        Patch(facecolor=get_scenario_color(rcp), edgecolor="none") for rcp in rcp_order
    ]
    scenario_handles.extend(
        [
            Patch(facecolor="grey", alpha=cfg["bar_alphas"]["2100"]),
            Patch(facecolor="grey", alpha=cfg["bar_alphas"]["2050"]),
        ]
    )
    scenario_labels = [RCP_LABELS[rcp] for rcp in rcp_order] + [
        "2100 (background)",
        "2050 (overlay)",
    ]

    # Model legend (hatches)
    model_handles = []
    for model in model_order:
        hatch = cfg["model_hatches"][model]
        if hatch:
            model_handles.append(
                Patch(facecolor="white", hatch=hatch, edgecolor="black")
            )
        else:
            model_handles.append(Patch(facecolor="grey", edgecolor="black"))

    legend1 = ax.legend(
        scenario_handles,
        scenario_labels,
        loc="upper right",
        fontsize=legend_cfg["fontsize"],
        title="Scenario",
        frameon=False,
        bbox_to_anchor=legend_cfg["scenario_bbox"],
        title_fontproperties={"weight": "bold", "size": legend_cfg["title_fontsize"]},
    )

    legend2 = ax.legend(
        model_handles,
        model_order,
        loc="upper right",
        fontsize=legend_cfg["fontsize"],
        title="Model",
        frameon=False,
        bbox_to_anchor=legend_cfg["model_bbox"],
        title_fontproperties={"weight": "bold", "size": legend_cfg["title_fontsize"]},
    )

    legend2._legend_box.align = "left"
    legend1._legend_box.align = "left"
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    # Axis formatting
    axis_cfg = cfg["axis"]
    ax.set_yticks(y_pos[::-1])
    ax.set_yticklabels(top_countries, fontsize=axis_cfg["ylabel_fontsize"])
    ax.set_xlabel(axis_cfg["xlabel"], fontsize=axis_cfg["xlabel_fontsize"])

    # Grid
    grid_cfg = cfg["grid"]
    ax.xaxis.set_minor_locator(plt.MultipleLocator(grid_cfg["minor_interval"]))
    ax.grid(
        True,
        axis="x",
        linestyle=":",
        alpha=grid_cfg["minor_alpha"],
        which="minor",
        zorder=0,
    )
    ax.grid(True, axis="x", linestyle=":", alpha=grid_cfg["major_alpha"])

    # Spines and ticks
    ax.tick_params(axis="y", length=0)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", which="major", pad=axis_cfg["y_tick_pad"])

    plt.tight_layout()
    plt.subplots_adjust(bottom=cfg["figure"]["bottom_margin"])

    if save_path:
        fig.savefig(save_path, dpi=cfg["figure"]["save_dpi"], bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_loss_distribution(
    result: DepreciationResult, ax: plt.Axes = None, save_path: Path = None
) -> plt.Figure:
    """
    Histogram of site-level value losses.

    Parameters
    ----------
    result : DepreciationResult
        Analysis result.
    ax : Axes, optional
        Matplotlib axes.
    save_path : Path, optional
        Save path.

    Returns
    -------
    Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    losses = result.gdf["value_loss"]
    losses_nonzero = losses[losses > 0]

    ax.hist(
        losses_nonzero,
        bins=50,
        color=get_scenario_color(result.scenario),
        alpha=0.7,
        edgecolor="white",
    )
    ax.set_xlabel("Value Loss ($)", fontsize=12)
    ax.set_ylabel("Number of Sites", fontsize=12)
    ax.set_title(
        f"Distribution of Site-Level Losses\n{result.scenario} | {result.model.name}",
        fontsize=14,
    )
    ax.set_yscale("log")

    # Add summary stats
    textstr = f"Mean: {format_currency(losses_nonzero.mean())}\nMedian: {format_currency(losses_nonzero.median())}\nTotal: {format_currency(losses_nonzero.sum())}"
    ax.text(
        0.95,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# INTERACTIVE (PLOTLY) PLOTS
# =============================================================================


def plot_choropleth_interactive(
    result: DepreciationResult,
    value_column: str = "value_loss",
    log_scale: bool = True,
    save_path: Path = None,
) -> go.Figure:
    """
    Interactive choropleth map of losses by country.

    Parameters
    ----------
    result : DepreciationResult
        Analysis result.
    value_column : str
        Column to visualize ("value_loss", "loss_fraction", "original_value").
    log_scale : bool
        Use logarithmic color scale.
    save_path : Path, optional
        Save as HTML.

    Returns
    -------
    Figure
        Plotly figure.
    """
    by_country = result.by_country.copy()

    # Get ISO codes
    iso_col = "iso_a3" if "iso_a3" in result.gdf.columns else None
    if iso_col:
        # Aggregate ISO codes
        country_col = result._get_country_column()
        iso_map = result.gdf.groupby(country_col)["iso_a3"].first()
        by_country["iso_a3"] = by_country[result._get_country_column()].map(iso_map)
    else:
        by_country["iso_a3"] = by_country.index  # Assume index is ISO code

    def _format_percent(v):
        return f"{v:.1f}%"

    # Prepare display values
    if value_column == "loss_fraction":
        z_values = by_country[value_column] * 100
        colorbar_title = "Loss (%)"
        hover_format = _format_percent
    else:
        z_values = by_country[value_column]
        colorbar_title = "Value Loss ($)"
        hover_format = format_currency

    # Apply log scale if requested
    if log_scale and value_column != "loss_fraction":
        z_safe = z_values.replace(0, np.nan)
        min_val = z_safe[z_safe > 0].min() if (z_safe > 0).any() else 1
        z_safe = z_safe.fillna(min_val)
        z_display = np.log10(z_safe)

        # Colorbar ticks
        min_log = int(np.floor(np.log10(min_val)))
        max_log = int(np.ceil(np.log10(z_safe.max())))
        tick_vals = list(range(min_log, max_log + 1))
        tick_text = [format_currency(10**v) for v in tick_vals]
    else:
        z_display = z_values
        tick_vals = None
        tick_text = None

    # Create hover text
    country_col = result._get_country_column()
    hover_text = [
        f"{row[country_col]}<br>"
        f"Loss: {hover_format(row[value_column])}<br>"
        f"Coral Change: {row['mean_coral_change'] * 100:.1f}pp"
        for _, row in by_country.iterrows()
    ]

    fig = go.Figure(
        data=go.Choropleth(
            locations=by_country["iso_a3"],
            z=z_display,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorscale="Reds",
            colorbar=dict(
                title=colorbar_title,
                tickvals=tick_vals,
                ticktext=tick_text,
            )
            if tick_vals
            else dict(title=colorbar_title),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Economic Losses by Country<br>{result.scenario} | {result.model.name}",
            x=0.5,
            y=0.95,
        ),
        geo=dict(
            showcoastlines=True,
            showland=True,
            landcolor="lightgray",
            projection_type="natural earth",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_model_comparison_interactive(
    models: List[DepreciationModel] = None, value: float = 100.0, save_path: Path = None
) -> go.Figure:
    """
    Interactive comparison of depreciation models.

    Parameters
    ----------
    models : list, optional
        DepreciationModel instances.
    value : float
        Base value.
    save_path : Path, optional
        Save as HTML.

    Returns
    -------
    Figure
        Plotly figure.
    """
    from .depreciation_models import CompoundModel, LinearModel

    if models is None:
        # models = [LinearModel(), CompoundModel(), TippingPointModel()]
        models = [LinearModel(), CompoundModel()]

    delta_cc_range = np.linspace(0, -1.0, 101)
    x_values = np.abs(delta_cc_range) * 100  # Percentage points

    fig = go.Figure()

    for model in models:
        remaining = model.calculate(delta_cc_range, value)
        loss_pct = 100 * (value - remaining) / value

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=loss_pct,
                name=model.name,
                mode="lines",
                line=dict(width=3),
                hovertemplate=f"{model.name}<br>Coral decrease: %{{x:.0f}}pp<br>Value loss: %{{y:.1f}}%<extra></extra>",
            )
        )

    fig.update_layout(
        title="Depreciation Model Comparison",
        xaxis_title="Coral Cover Decrease (percentage points)",
        yaxis_title="Value Loss (%)",
        hovermode="x unified",
        template="plotly_white",
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# REPORT GENERATION
# =============================================================================


def generate_figure_set(
    result: DepreciationResult,
    output_dir: Path,
    prefix: str = "",
    formats: List[str] = None,
) -> Dict[str, Path]:
    """
    Generate a complete set of figures for a result.

    Parameters
    ----------
    result : DepreciationResult
        Analysis result.
    output_dir : Path
        Output directory.
    prefix : str
        Filename prefix.
    formats : list, optional
        Output formats. Default: ["png", "html"].

    Returns
    -------
    dict
        Mapping of figure names to file paths.
    """
    if formats is None:
        formats = ["png", "html"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_slug = utils.sanitize_filename(result.scenario)
    model_slug = utils.sanitize_filename(result.model.name)
    base_name = f"{prefix}{result.value_type}_{scenario_slug}_{model_slug}"

    saved_files = {}

    # Static plots (PNG)
    if "png" in formats:
        # Country losses
        fig = plot_country_losses(result, top_n=20)
        path = output_dir / f"{base_name}_country_losses.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files["country_losses"] = path

        # Loss distribution
        fig = plot_loss_distribution(result)
        path = output_dir / f"{base_name}_loss_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files["loss_distribution"] = path

    # Interactive plots (HTML)
    if "html" in formats:
        fig = plot_choropleth_interactive(result)
        path = output_dir / f"{base_name}_choropleth.html"
        fig.write_html(path)
        saved_files["choropleth_interactive"] = path

    print(f"✓ Generated {len(saved_files)} figures for {base_name}")

    return saved_files


# =============================================================================
# VERIFICATION PLOTS (from original notebook)
# =============================================================================


def plot_gdp_percentage_bar(
    by_country_df,
    gdp_pct_column: str = "reef_tourism_gdp_as_pct_of_national_gdp",
    country_column: str = "country",
    top_n: int = 20,
    save_path: Path = None,
) -> plt.Figure:
    """
    Horizontal bar chart of reef tourism as % of national GDP.

    Replicates the notebook verification plot.

    Parameters
    ----------
    by_country_df : DataFrame
        Aggregated data by country with GDP percentage column.
    gdp_pct_column : str
        Column name for GDP percentage.
    country_column : str
        Column name for country.
    top_n : int
        Number of countries to show.
    save_path : Path, optional
        Save path.

    Returns
    -------
    Figure
    """
    df = by_country_df.copy()
    df = df.sort_values(by=gdp_pct_column, ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))

    ax.barh(
        df[country_column].values[::-1],
        df[gdp_pct_column].values[::-1],
        color="#3498db",
        alpha=0.8,
    )
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlabel("Reef tourism revenue as % of national GDP", fontsize=12)
    ax.set_title(f"Top {top_n} Countries by Reef Tourism GDP Contribution", fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_gdp_percentage_choropleth(
    by_country_df,
    gdp_pct_column: str = "reef_tourism_gdp_as_pct_of_national_gdp",
    country_column: str = "country",
    iso_column: str = None,
    log_scale: bool = True,
    save_path: Path = None,
) -> go.Figure:
    """
    Interactive choropleth of reef tourism as % of national GDP.

    Replicates the notebook verification plot.

    Parameters
    ----------
    by_country_df : DataFrame
        Aggregated data by country.
    gdp_pct_column : str
        Column name for GDP percentage.
    country_column : str
        Column name for country.
    iso_column : str, optional
        Column name for ISO codes. If None, uses index.
    log_scale : bool
        Use logarithmic color scale.
    save_path : Path, optional
        Save as HTML.

    Returns
    -------
    Figure
        Plotly figure.
    """
    import plotly.express as px

    df = by_country_df.copy()

    # Get ISO codes
    if iso_column and iso_column in df.columns:
        iso_codes = df[iso_column]
    elif df.index.name == "iso_a3" or (
        df.index.dtype == "object" and df.index.str.len().min() == 3
    ):
        iso_codes = df.index
    else:
        raise ValueError(
            "Cannot find ISO codes. Provide iso_column or set index to ISO codes."
        )

    z_data = df[gdp_pct_column].copy()

    if log_scale:
        # Handle zeros for log scale
        z_safe = z_data.replace(0, np.nan)
        min_val = z_safe[z_safe > 0].min() if (z_safe > 0).any() else 1e-2
        z_safe = z_safe.fillna(min_val)
        z_display = np.log10(z_safe)

        # Colorbar ticks
        tick_vals = np.log10([1e-2, 0.1, 1, 10, 100])
        tick_text = [f"{v:.2g}%" for v in [1e-2, 0.1, 1, 10, 100]]
        zmin = np.log10(min_val)
        zmax = np.log10(z_safe.max())
    else:
        z_display = z_data
        tick_vals = None
        tick_text = None
        zmin = None
        zmax = None

    # Hover text
    hover_text = [
        f"{row[country_column]}<br>Contribution to GDP: {row[gdp_pct_column]:.2f}%"
        for _, row in df.iterrows()
    ]

    # Build colorbar config
    if tick_vals is not None:
        colorbar_config = dict(
            title="Contribution to GDP (%)",
            tickvals=tick_vals,
            ticktext=tick_text,
            orientation="h",
            x=0.5,
            y=-0.2,
            len=0.6,
        )
    else:
        colorbar_config = dict(title="Contribution to GDP (%)")

    fig = go.Figure(
        data=go.Choropleth(
            locations=iso_codes,
            z=z_display,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorscale=px.colors.sequential.Plasma,
            colorbar=colorbar_config,
            zmin=zmin,
            zmax=zmax,
        )
    )

    fig.update_layout(
        geo=dict(showcoastlines=True, showland=True),
        title=dict(
            text="Contribution of reefs to GDP by country"
            + ("<br>(logarithmic color scale)" if log_scale else ""),
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        autosize=True,
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_total_revenue_choropleth(
    by_country_df,
    value_column: str = "approx_price_corrected",
    country_column: str = "country",
    iso_column: str = None,
    save_path: Path = None,
) -> go.Figure:
    """
    Interactive choropleth of total reef-associated tourism revenue.

    Replicates the notebook verification plot.

    Parameters
    ----------
    by_country_df : DataFrame
        Aggregated data by country.
    value_column : str
        Column name for tourism value.
    country_column : str
        Column name for country.
    iso_column : str, optional
        Column name for ISO codes.
    save_path : Path, optional
        Save as HTML.

    Returns
    -------
    Figure
        Plotly figure.
    """
    import plotly.express as px

    df = by_country_df.copy()

    # Get ISO codes
    if iso_column and iso_column in df.columns:
        iso_codes = df[iso_column]
    elif df.index.name == "iso_a3" or (
        df.index.dtype == "object" and df.index.str.len().min() == 3
    ):
        iso_codes = df.index
    else:
        raise ValueError("Cannot find ISO codes.")

    z_data = df[value_column]

    # Tick formatting
    max_val = z_data.max()
    tickvals = np.arange(0, max_val, 5e8)
    ticktext = [format_currency(v, 0) for v in tickvals]

    # Hover text
    hover_text = [
        f"{row[country_column]}<br>Contribution to GDP: {format_currency(row[value_column])}"
        for _, row in df.iterrows()
    ]

    fig = go.Figure(
        data=go.Choropleth(
            locations=iso_codes,
            z=z_data,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorscale=px.colors.sequential.Plasma,
            colorbar=dict(
                title="Total reef-associated tourism revenue",
                tickvals=tickvals,
                ticktext=ticktext,
                orientation="h",
                x=0.5,
                y=-0.2,
                len=0.6,
            ),
        )
    )

    fig.update_layout(
        geo=dict(showcoastlines=True, showland=True),
        title=dict(
            text="Contribution of reefs to GDP by country",
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        autosize=True,
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_tourism_value_bins_map(
    tourism_gdf,
    bin_column: str = "bin_global",
    bbox: tuple = None,
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot tourism value bins as a choropleth map (like the Caribbean example).

    Parameters
    ----------
    tourism_gdf : GeoDataFrame
        Tourism data with bin values.
    bin_column : str
        Column containing bin values (0-10).
    bbox : tuple, optional
        Bounding box (minx, miny, maxx, maxy) to zoom to.
    save_path : Path, optional
        Save path.

    Returns
    -------
    Figure
    """
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap, Normalize

    # Price bin labels
    prices_key = [
        "no value",
        "up to 4000",
        "4001-8000",
        "8001-12000",
        "12001-24000",
        "24001-44000",
        "44001-92000",
        "92001-172000",
        "172001-352000",
        "352001-908000",
        ">908000",
    ]

    # Custom colormap to match Ocean Wealth display
    colormap = [
        "#828282",
        "#2892c8",
        "#74b474",
        "#52bf81",
        "#59d95e",
        "#58f230",
        "#f6e058",
        "#e6ac3e",
        "#d57726",
        "#bf4713",
        "#730000",
    ]
    price_cmap = ListedColormap(colormap)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter to bbox if provided (zoom to see details)
    if bbox:
        from shapely.geometry import box

        bbox_geom = box(*bbox)
        gdf_plot = tourism_gdf[tourism_gdf.geometry.intersects(bbox_geom)]
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])
    else:
        gdf_plot = tourism_gdf

    gdf_plot.plot(
        ax=ax,
        column=bin_column,
        cmap=price_cmap,
        legend=False,
        linewidth=0.2,
        edgecolor=None,
    )

    # Colorbar
    norm = Normalize(vmin=gdf_plot[bin_column].min(), vmax=gdf_plot[bin_column].max())
    sm = cm.ScalarMappable(cmap=price_cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(
        sm,
        ax=ax,
        label="Tourism value bin",
        orientation="horizontal",
        pad=0.1,
        aspect=50,
        ticks=np.linspace(0.5, 9.5, len(prices_key)),
    )
    cb.set_ticklabels(prices_key)
    cb.ax.tick_params(rotation=45)

    ax.set_title("Tourism Value by Bin", fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_spatial_distribution(
    gdf,
    plot_column: str = "bin_global",
    bbox: tuple = None,
    title: str = None,
    save_path: Path = None,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    transform: ccrs.CRS = None,
    explode_factor: float = 1.0,
    vmin: float = None,
    vmax: float = None,
    logarithmic_cbar: bool = False,
    config: "SpatialPlotConfig" = None,
    central_longitude: float = None,
    extent: tuple = None,
    show_scalebar: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a spatial distribution of a column in a GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Data with values to plot.
    plot_column : str
        Column containing values to plot.
    bbox : tuple, optional
        Bounding box (minx, miny, maxx, maxy) to zoom to. Deprecated: use extent in config.
    title : str, optional
        Plot title. Overrides config.title if provided.
    save_path : Path, optional
        Save path.
    fig : Figure, optional
        Matplotlib figure to plot on.
    ax : Axes, optional
        Matplotlib axes to plot on.
    transform : ccrs.CRS, optional
        Cartopy coordinate reference system for the plot. Deprecated: use config.
    explode_factor : float, optional
        Factor to scale polygons from their centroids. Deprecated: use config.explode_factor.
    vmin : float, optional
        Minimum value for colorbar. Deprecated: use config.vmin.
    vmax : float, optional
        Maximum value for colorbar. Deprecated: use config.vmax.
    logarithmic_cbar : bool, optional
        Use logarithmic colorbar. Deprecated: use config.logarithmic_cbar.
    config : SpatialPlotConfig, optional
        Configuration object for plot formatting. If provided, overrides individual parameters.
    central_longitude : float, optional
        Central longitude for map projection. Overrides config if provided.
    extent : tuple, optional
        Map extent (x0, x1, y0, y1). Overrides config if provided.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes objects.
    """
    import matplotlib.cm as cm
    from matplotlib.colors import LogNorm, Normalize
    from shapely.affinity import scale, translate

    from src.plots import plot_utils
    from src.plots.plot_config import PAPER_SPATIAL_CONFIG
    from src.plots.plot_utils import transform_coordinates_for_central_longitude

    # Create or merge config
    if config is None:
        config = PAPER_SPATIAL_CONFIG.copy()

    # print(f"config: {config}")
    config = plot_utils.override_config_with_kwargs(
        config,
        central_longitude=central_longitude,
        extent=extent,
        title=title,
        explode_factor=explode_factor,
        vmin=vmin,
        vmax=vmax,
        logarithmic_cbar=logarithmic_cbar,
        map_proj=transform,
        show_scalebar=show_scalebar,
    )
    # print(f"config: {config}")

    # Create figure/axes if not provided
    if fig is None or ax is None:
        fig, ax = plot_utils.generate_geo_axis(
            figsize=config.figsize,
            dpi=config.dpi,
            central_longitude=config.central_longitude,
            central_latitude=config.central_latitude,
            map_proj=config.map_proj,
            config=config,
        )

    # Format axes
    plot_utils.format_geo_axes(ax, config=config)

    # # This works and sorts out the colourmap issue (at least, it shows the variety of colours and adjusts the colourbar)
    # if bbox:
    #     from shapely.geometry import box

    #     bbox_geom = box(*bbox)
    #     gdf_plot = gdf[gdf.geometry.intersects(bbox_geom)].copy()
    # else:
    #     gdf_plot = gdf.copy()
    gdf_plot = gdf.copy()

    # Transform coordinates if central_longitude is set
    if config.central_longitude is not None and config.transform_coords:
        gdf_plot = transform_coordinates_for_central_longitude(
            gdf_plot, config.central_longitude
        )

    # Apply explode factor if specified
    if config.explode_factor != 1.0:

        def scale_from_centroid(geom):
            """Scale geometry from its centroid by explode_factor"""
            if geom.is_empty:
                return geom

            # Get centroid
            centroid = geom.centroid
            cx, cy = centroid.x, centroid.y

            # Translate to origin, scale, then translate back
            geom_translated = translate(geom, xoff=-cx, yoff=-cy)
            geom_scaled = scale(
                geom_translated,
                xfact=config.explode_factor,
                yfact=config.explode_factor,
                origin=(0, 0),
            )
            geom_final = translate(geom_scaled, xoff=cx, yoff=cy)

            return geom_final

        gdf_plot["geometry"] = gdf_plot.geometry.apply(scale_from_centroid)

    # Get colormap
    cmap = plt.get_cmap(config.cmap)

    # Get transform for plotting
    plot_transform = config.get_projection()

    # Colorbar normalization
    vmin_val = gdf_plot[plot_column].min() if config.vmin is None else config.vmin
    vmax_val = gdf_plot[plot_column].max() if config.vmax is None else config.vmax

    if config.logarithmic_cbar:
        if vmin_val <= 0:
            print("Warning: vmin is less than or equal to 0, setting vmin to 1e-10")
            vmin_val = 1e-10
            # snap values less than vmin_val to vmin_val
            gdf_plot[plot_column] = gdf_plot[plot_column].clip(lower=vmin_val)
        print("Using logarithmic colorbar")
        norm = LogNorm(vmin=vmin_val, vmax=vmax_val)
    else:
        print(f"vmin_val: {vmin_val:.3e}, vmax_val: {vmax_val:.3e}")
        norm = Normalize(vmin=vmin_val, vmax=vmax_val)

    # Plot
    gdf_plot.plot(
        ax=ax,
        column=plot_column,
        cmap=cmap,
        legend=False,
        linewidth=config.edgewidth,
        edgecolor=config.edgecolor,
        transform=plot_transform,
        norm=norm,
    )

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(
        sm,
        ax=ax,
        label=plot_column,
        orientation=config.cbar_orientation,
        pad=config.cbar_pad,
        aspect=config.cbar_aspect,
    )

    # Labels and title
    title_text = (
        config.title
        if config.title is not None
        else f"Spatial Distribution of '{plot_column}'"
    )
    ax.set_title(title_text, fontsize=config.title_fontsize)

    xlabel = config.xlabel if config.xlabel is not None else "Longitude"
    ylabel = config.ylabel if config.ylabel is not None else "Latitude"
    ax.set_xlabel(xlabel, fontsize=config.label_fontsize)
    ax.set_ylabel(ylabel, fontsize=config.label_fontsize)

    # Add scalebar if enabled
    if config.show_scalebar:
        plot_utils.add_scalebar(
            ax=ax,
            length=config.scalebar_length,
            location=config.scalebar_location,
            units=config.scalebar_units,
            segments=config.scalebar_segments,
            linewidth=config.scalebar_linewidth,
            fontsize=config.scalebar_fontsize,
            color=config.scalebar_color,
            tick_rotation=config.scalebar_tick_rotation,
            frame=config.scalebar_frame,
        )

    if config.tight_layout:
        plt.tight_layout()

    if save_path:
        fig.savefig(
            save_path,
            dpi=config.save_dpi,
            bbox_inches="tight",
            format=config.save_format,
        )
        print(f"Saved: {save_path}")

    return fig, ax


def plot_spatial_distribution_interactive(
    gdf,
    plot_column: str = "bin_global",
    bbox: tuple = None,
    save_path: Path = None,
) -> "folium.Map":
    """
    Create an interactive map showing the spatial distribution of a column in a GeoDataFrame,
    using folium for high-resolution, interactive mapping.

    Parameters
    ----------
    gdf : GeoDataFrame
        Data with values to plot.
    plot_column : str
        Column containing values to plot.
    bbox : tuple, optional
        Bounding box (minx, miny, maxx, maxy) to zoom to.
    save_path : Path, optional
        Path to save the HTML map.

    Returns
    -------
    folium.Map
        The interactive folium map.
    """
    import branca
    import folium
    import matplotlib
    import numpy as np

    # Filter by bounding box if provided
    if bbox:
        from shapely.geometry import box

        bbox_geom = box(*bbox)
        gdf_plot = gdf[gdf.geometry.intersects(bbox_geom)].copy()
        gdf_plot = gdf_plot[gdf_plot[plot_column].notnull()]
    else:
        gdf_plot = gdf.copy()
        gdf_plot = gdf_plot[gdf_plot[plot_column].notnull()]

    # Calculate bounds
    if bbox:
        minx, miny, maxx, maxy = bbox
    else:
        minx, miny, maxx, maxy = gdf_plot.total_bounds

    # Center of the map
    center = [(miny + maxy) / 2, (minx + maxx) / 2]

    # Choose colormap to match appearance with matplotlib's "turbo"
    cmap = matplotlib.cm.get_cmap("turbo")
    values = gdf_plot[plot_column].values.astype(float)
    vmin, vmax = np.nanmin(values), np.nanmax(values)

    # Create a branca colormap (for folium coloring)
    colormap = branca.colormap.LinearColormap(
        [matplotlib.colors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, 256)],
        vmin=vmin,
        vmax=vmax,
    )
    colormap.caption = plot_column

    # Reproject to WGS84 if not already
    if gdf_plot.crs is not None and gdf_plot.crs.to_epsg() != 4326:
        gdf_plot = gdf_plot.to_crs(epsg=4326)

    # Create the folium map
    m = folium.Map(location=center, zoom_start=7, tiles="cartodbpositron")

    # Add features at reasonable geojson resolution (for high-res display)
    # If there are lots of polygons, simplify to a small tolerance for performance,
    # but we'll use a small value so detail is high.
    gdf_plot = gdf_plot.copy()
    gdf_plot["geometry"] = gdf_plot["geometry"].simplify(
        tolerance=0.0003, preserve_topology=True
    )

    def style_function(feature):
        val = feature["properties"][plot_column]
        if val is not None:
            color = colormap(val)
        else:
            color = "#999999"
        return {
            "fillOpacity": 0.7,
            "weight": 0.2,
            "color": color,
            "fillColor": color,
        }

    folium.GeoJson(
        gdf_plot,
        name="spatial",
        style_function=style_function,
        highlight_function=lambda x: {"weight": 2, "color": "yellow"},
        tooltip=folium.GeoJsonTooltip(fields=[plot_column]),
    ).add_to(m)

    colormap.add_to(m)

    m.fit_bounds([[miny, minx], [maxy, maxx]])

    if save_path:
        save_path = str(save_path)
        m.save(save_path)
        print(f"Interactive map saved: {save_path}")

    return m


def generate_verification_plots(
    tourism_gdf,
    by_country_df,
    output_dir: Path,
    gdp_pct_column: str = "reef_tourism_gdp_as_pct_of_national_gdp",
    value_column: str = "approx_price_corrected",
) -> Dict[str, Path]:
    """
    Generate all verification plots from the original notebook.

    Parameters
    ----------
    tourism_gdf : GeoDataFrame
        Tourism data with polygon values.
    by_country_df : DataFrame
        Aggregated data by country with GDP percentage.
    output_dir : Path
        Output directory.
    gdp_pct_column : str
        Column for GDP percentage.
    value_column : str
        Column for tourism value.

    Returns
    -------
    dict
        Mapping of plot names to file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # 1. GDP percentage bar chart
    fig = plot_gdp_percentage_bar(
        by_country_df, gdp_pct_column=gdp_pct_column, top_n=20
    )
    path = output_dir / "verification_gdp_pct_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_files["gdp_pct_bar"] = path

    # 2. GDP percentage choropleth (log scale)
    fig = plot_gdp_percentage_choropleth(
        by_country_df,
        gdp_pct_column=gdp_pct_column,
        log_scale=True,
    )
    path = output_dir / "verification_gdp_pct_choropleth.html"
    fig.write_html(path)
    saved_files["gdp_pct_choropleth"] = path

    # 3. Total revenue choropleth
    fig = plot_total_revenue_choropleth(
        by_country_df,
        value_column=value_column,
    )
    path = output_dir / "verification_total_revenue_choropleth.html"
    fig.write_html(path)
    saved_files["total_revenue_choropleth"] = path

    # 4. Tourism bins map (Caribbean zoom)
    if "bin_global" in tourism_gdf.columns:
        fig = plot_tourism_value_bins_map(
            tourism_gdf,
            bbox=(-83.5, 24, -80, 25),  # Caribbean
        )
        path = output_dir / "verification_tourism_bins_caribbean.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files["tourism_bins_caribbean"] = path

    print(f"✓ Generated {len(saved_files)} verification plots")

    return saved_files


# =============================================================================
# DEPRECIATION AS % OF GDP PLOTS
# =============================================================================


def plot_loss_as_gdp_pct_bar(
    result,
    gdp_data,
    top_n: int = 20,
    save_path: Path = None,
) -> plt.Figure:
    """
    Bar chart of projected value loss as % of national GDP.

    Parameters
    ----------
    result : DepreciationResult
        Analysis result with value_loss by country.
    gdp_data : DataFrame
        GDP data with 'iso_a3' and 'gdp' columns.
    top_n : int
        Number of countries to show.
    save_path : Path, optional
        Save path.

    Returns
    -------
    Figure
    """
    by_country = result.by_country.copy()

    # Get country column
    country_col = (
        "country" if "country" in by_country.columns else by_country.columns[0]
    )

    # Get ISO codes from result
    iso_col = "iso_a3" if "iso_a3" in by_country.columns else None
    if iso_col is None and "iso_a3" in result.gdf.columns:
        # Map from gdf
        iso_map = result.gdf.groupby(result._get_country_column())["iso_a3"].first()
        by_country["iso_a3"] = by_country[country_col].map(iso_map)
        iso_col = "iso_a3"

    if iso_col is None:
        warnings.warn("Cannot find ISO codes for GDP matching")
        return None

    # Map GDP
    gdp_map = gdp_data.set_index("iso_a3")["gdp"]
    by_country["national_gdp"] = by_country[iso_col].map(gdp_map)
    by_country["loss_as_gdp_pct"] = (
        100 * by_country["value_loss"] / by_country["national_gdp"]
    )

    # Filter and sort
    df = by_country.dropna(subset=["loss_as_gdp_pct"])
    df = df.sort_values("loss_as_gdp_pct", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))

    color = get_scenario_color(result.scenario)
    ax.barh(
        df[country_col].values[::-1],
        df["loss_as_gdp_pct"].values[::-1],
        color=color,
        alpha=0.8,
    )

    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlabel("Projected Value Loss as % of National GDP", fontsize=12)
    ax.set_title(
        f"Top {top_n} Countries: Tourism Loss as % of GDP\n{result.scenario} | {result.model.name}",
        fontsize=14,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_loss_as_gdp_pct_choropleth(
    result,
    gdp_data,
    log_scale: bool = True,
    save_path: Path = None,
) -> go.Figure:
    """
    Interactive choropleth of value loss as % of national GDP.

    Parameters
    ----------
    result : DepreciationResult
        Analysis result.
    gdp_data : DataFrame
        GDP data with 'iso_a3' and 'gdp' columns.
    log_scale : bool
        Use logarithmic color scale.
    save_path : Path, optional
        Save as HTML.

    Returns
    -------
    Figure
        Plotly figure.
    """

    by_country = result.by_country.copy()

    # Get country column
    country_col = (
        "country" if "country" in by_country.columns else by_country.columns[0]
    )

    # Get ISO codes
    iso_col = "iso_a3" if "iso_a3" in by_country.columns else None
    if iso_col is None and "iso_a3" in result.gdf.columns:
        iso_map = result.gdf.groupby(result._get_country_column())["iso_a3"].first()
        by_country["iso_a3"] = by_country[country_col].map(iso_map)
        iso_col = "iso_a3"

    if iso_col is None:
        warnings.warn("Cannot find ISO codes for GDP matching")
        return None

    # Map GDP and calculate loss as % of GDP
    gdp_map = gdp_data.set_index("iso_a3")["gdp"]
    by_country["national_gdp"] = by_country[iso_col].map(gdp_map)
    by_country["loss_as_gdp_pct"] = (
        100 * by_country["value_loss"] / by_country["national_gdp"]
    )

    # Filter NaN
    df = by_country.dropna(subset=["loss_as_gdp_pct"])

    z_data = df["loss_as_gdp_pct"]

    if log_scale:
        z_safe = z_data.replace(0, np.nan)
        min_val = z_safe[z_safe > 0].min() if (z_safe > 0).any() else 1e-4
        z_safe = z_safe.fillna(min_val)
        z_display = np.log10(z_safe)

        tick_vals = np.log10([1e-4, 1e-3, 1e-2, 0.1, 1, 10])
        tick_text = ["0.0001%", "0.001%", "0.01%", "0.1%", "1%", "10%"]
        zmin = np.log10(min_val)
        zmax = np.log10(z_safe.max())
    else:
        z_display = z_data
        tick_vals = None
        tick_text = None
        zmin = None
        zmax = None

    # Hover text
    hover_text = [
        f"{row[country_col]}<br>"
        f"Value Loss: {format_currency(row['value_loss'])}<br>"
        f"Loss as % of GDP: {row['loss_as_gdp_pct']:.4f}%"
        for _, row in df.iterrows()
    ]

    # Build colorbar
    if tick_vals is not None:
        colorbar_config = dict(
            title="Loss as % of GDP",
            tickvals=tick_vals,
            ticktext=tick_text,
            orientation="h",
            x=0.5,
            y=-0.15,
            len=0.6,
        )
    else:
        colorbar_config = dict(title="Loss as % of GDP")

    fig = go.Figure(
        data=go.Choropleth(
            locations=df[iso_col],
            z=z_display,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorscale="Reds",
            colorbar=colorbar_config,
            zmin=zmin,
            zmax=zmax,
        )
    )

    fig.update_layout(
        geo=dict(showcoastlines=True, showland=True),
        title=dict(
            text=f"Projected Tourism Value Loss as % of National GDP<br>{result.scenario} | {result.model.name}"
            + ("<br>(logarithmic scale)" if log_scale else ""),
            x=0.5,
            y=0.95,
            xanchor="center",
            yanchor="top",
        ),
        margin=dict(l=0, r=0, t=80, b=60),
    )

    if save_path:
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# TRAJECTORY AND CUMULATIVE IMPACT PLOTS
# =============================================================================


def plot_coral_cover_trajectories(
    results: dict,
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot coral cover trajectories over time for multiple scenarios.

    Parameters
    ----------
    results : dict
        Mapping of scenario names to CumulativeImpactResult.
    save_path : Path, optional
        Save path.

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color and style mappings
    scenario_colors = {
        "RCP45": "#3498db",
        "RCP85": "#e74c3c",
        "rcp45": "#3498db",
        "rcp85": "#e74c3c",
    }
    method_styles = {"linear": "-", "exponential": "--"}

    for key, result in results.items():
        traj = result.trajectory
        scenario = traj.scenario.upper() if len(traj.scenario) <= 5 else traj.scenario

        # Determine color based on scenario
        color = "#7f8c8d"  # default gray
        for sc, c in scenario_colors.items():
            if sc.lower() in key.lower():
                color = c
                break

        style = method_styles.get(traj.interpolation_method, "-")

        label = f"{scenario} ({traj.interpolation_method.title()})"
        ax.plot(
            traj.years,
            traj.covers * 100,
            style,
            color=color,
            linewidth=2,
            label=label,
            alpha=0.8,
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Coral Cover (%)", fontsize=12)
    ax.set_title("Projected Coral Cover Trajectories", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=2017)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_annual_value_trajectories(
    results: dict,
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot annual economic value trajectories over time.

    Parameters
    ----------
    results : dict
        Mapping of scenario names to CumulativeImpactResult.
    save_path : Path, optional
        Save path.

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    scenario_colors = {
        "RCP45": "#3498db",
        "RCP85": "#e74c3c",
        "rcp45": "#3498db",
        "rcp85": "#e74c3c",
    }
    method_styles = {"linear": "-", "exponential": "--"}

    baseline_value = None

    for key, result in results.items():
        traj = result.trajectory
        scenario = traj.scenario.upper() if len(traj.scenario) <= 5 else traj.scenario

        if baseline_value is None:
            baseline_value = result.baseline_value

        color = "#7f8c8d"
        for sc, c in scenario_colors.items():
            if sc.lower() in key.lower():
                color = c
                break

        style = method_styles.get(traj.interpolation_method, "-")
        label = f"{scenario} ({traj.interpolation_method.title()})"

        ax.plot(
            traj.years,
            result.annual_values / 1e9,
            style,
            color=color,
            linewidth=2,
            label=label,
            alpha=0.8,
        )

    # Add baseline reference
    if baseline_value:
        ax.axhline(
            y=baseline_value / 1e9,
            color="green",
            linestyle=":",
            linewidth=1.5,
            label="Baseline Value",
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Annual Value ($ Billion)", fontsize=12)
    ax.set_title(
        "Projected Annual Economic Value Trajectories", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=2017)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_cumulative_loss_trajectories(
    results: dict,
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot cumulative economic losses over time.

    Parameters
    ----------
    results : dict
        Mapping of scenario names to CumulativeImpactResult.
    save_path : Path, optional
        Save path.

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    scenario_colors = {
        "RCP45": "#3498db",
        "RCP85": "#e74c3c",
        "rcp45": "#3498db",
        "rcp85": "#e74c3c",
    }
    method_styles = {"linear": "-", "exponential": "--"}

    for key, result in results.items():
        traj = result.trajectory
        scenario = traj.scenario.upper() if len(traj.scenario) <= 5 else traj.scenario

        color = "#7f8c8d"
        for sc, c in scenario_colors.items():
            if sc.lower() in key.lower():
                color = c
                break

        style = method_styles.get(traj.interpolation_method, "-")
        label = f"{scenario} ({traj.interpolation_method.title()})"

        ax.plot(
            traj.years,
            result.cumulative_losses / 1e12,
            style,
            color=color,
            linewidth=2,
            label=label,
            alpha=0.8,
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Cumulative Loss ($ Trillion)", fontsize=12)
    ax.set_title(
        "Cumulative Economic Losses Over Time\n(Running total of annual losses)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=2017)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_annual_loss_trajectories(
    results: dict,
    save_path: Path = None,
) -> plt.Figure:
    """
    Plot annual loss rate trajectories over time.

    Parameters
    ----------
    results : dict
        Mapping of scenario names to CumulativeImpactResult.
    save_path : Path, optional
        Save path.

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    scenario_colors = {
        "RCP45": "#3498db",
        "RCP85": "#e74c3c",
        "rcp45": "#3498db",
        "rcp85": "#e74c3c",
    }
    method_styles = {"linear": "-", "exponential": "--"}

    for key, result in results.items():
        traj = result.trajectory
        scenario = traj.scenario.upper() if len(traj.scenario) <= 5 else traj.scenario

        color = "#7f8c8d"
        for sc, c in scenario_colors.items():
            if sc.lower() in key.lower():
                color = c
                break

        style = method_styles.get(traj.interpolation_method, "-")
        label = f"{scenario} ({traj.interpolation_method.title()})"

        ax.plot(
            traj.years,
            result.annual_losses / 1e9,
            style,
            color=color,
            linewidth=2,
            label=label,
            alpha=0.8,
        )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Annual Loss ($ Billion/year)", fontsize=12)
    ax.set_title(
        "Annual Economic Loss Rate Over Time\n(Loss compared to baseline each year)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=2017)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_trajectory_comparison_interactive(
    results: dict,
    save_path: Path = None,
) -> go.Figure:
    """
    Interactive plot comparing trajectories with subplots.

    Parameters
    ----------
    results : dict
        Mapping of scenario names to CumulativeImpactResult.
    save_path : Path, optional
        Save as HTML.

    Returns
    -------
    Figure
        Plotly figure.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Coral Cover (%)",
            "Annual Value ($ Billion)",
            "Annual Loss ($ Billion/year)",
            "Cumulative Loss ($ Trillion)",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    scenario_colors = {
        "RCP45": "#3498db",
        "RCP85": "#e74c3c",
        "rcp45": "#3498db",
        "rcp85": "#e74c3c",
    }
    method_dashes = {"linear": "solid", "exponential": "dash"}

    for key, result in results.items():
        traj = result.trajectory
        scenario = traj.scenario.upper() if len(traj.scenario) <= 5 else traj.scenario

        color = "#7f8c8d"
        for sc, c in scenario_colors.items():
            if sc.lower() in key.lower():
                color = c
                break

        dash = method_dashes.get(traj.interpolation_method, "solid")
        name = f"{scenario} ({traj.interpolation_method.title()})"

        # Coral cover
        fig.add_trace(
            go.Scatter(
                x=traj.years,
                y=traj.covers * 100,
                mode="lines",
                name=name,
                line=dict(color=color, dash=dash, width=2),
                legendgroup=key,
                hovertemplate=f"{name}<br>Year: %{{x}}<br>Cover: %{{y:.1f}}%<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Annual value
        fig.add_trace(
            go.Scatter(
                x=traj.years,
                y=result.annual_values / 1e9,
                mode="lines",
                name=name,
                line=dict(color=color, dash=dash, width=2),
                legendgroup=key,
                showlegend=False,
                hovertemplate=f"{name}<br>Year: %{{x}}<br>Value: $%{{y:.1f}}B<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Annual loss
        fig.add_trace(
            go.Scatter(
                x=traj.years,
                y=result.annual_losses / 1e9,
                mode="lines",
                name=name,
                line=dict(color=color, dash=dash, width=2),
                legendgroup=key,
                showlegend=False,
                hovertemplate=f"{name}<br>Year: %{{x}}<br>Loss: $%{{y:.1f}}B/year<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Cumulative loss
        fig.add_trace(
            go.Scatter(
                x=traj.years,
                y=result.cumulative_losses / 1e12,
                mode="lines",
                name=name,
                line=dict(color=color, dash=dash, width=2),
                legendgroup=key,
                showlegend=False,
                hovertemplate=f"{name}<br>Year: %{{x}}<br>Cumulative: $%{{y:.2f}}T<extra></extra>",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title=dict(
            text="Coral Cover and Economic Impact Trajectories",
            x=0.5,
            y=0.98,
            xanchor="center",
        ),
        height=700,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
    )

    fig.update_xaxes(title_text="Year")

    if save_path:
        fig.write_html(save_path)
        print(f"Saved: {save_path}")

    return fig


def plot_gdp_impact_scenario_comparison(
    results,
    gdp_data,
    top_n: int = 15,
    save_path: Path = None,
) -> plt.Figure:
    import matplotlib.colors as mcolors

    cfg = SCENARIO_COMPARISON_CONFIG

    # --- Extract results ---
    relevant_results = list(results.results.items())
    if len(relevant_results) < 2:
        warnings.warn("Need at least 2 results to compare scenarios")
        return None

    # GDP lookup
    gdp_map = gdp_data.set_index("iso_a3")["gdp"]

    # --- Collate results {(rcp, model, year): by_country_df} ---
    scenario_dict = {}
    for _, result in relevant_results:
        rcp, year, model = parse_scenario(result.scenario, result.model.name)
        byc = result.by_country.copy()

        # Attach ISO codes if needed
        if "iso_a3" not in byc.columns and "iso_a3" in result.gdf.columns:
            iso_map = result.gdf.groupby(result._get_country_column())["iso_a3"].first()
            byc["iso_a3"] = byc[result._get_country_column()].map(iso_map)

        byc["national_gdp"] = byc["iso_a3"].map(gdp_map)
        byc["loss_as_gdp_pct"] = 100 * byc["value_loss"] / byc["national_gdp"]

        scenario_dict[(rcp.upper(), model, year)] = byc

    # --- Determine top countries (worst-case RCP85 / 2100) ---
    model_order = cfg["model_order"]
    rcp_order = cfg["rcp_order"]

    worst_key = ("RCP85", model_order[-1], "2100")
    worst_case = scenario_dict.get(worst_key, list(scenario_dict.values())[0])

    country_col = (
        "country" if "country" in worst_case.columns else worst_case.columns[0]
    )

    worst_case = worst_case.dropna(subset=["loss_as_gdp_pct"])
    top_countries = worst_case.nlargest(top_n, "loss_as_gdp_pct")[country_col].tolist()

    # --- Prepare data: {rcp: {model: {year: values}}} ---
    data = {rcp: {m: {} for m in model_order} for rcp in rcp_order}
    for (rcp, model, year), df in scenario_dict.items():
        if rcp not in rcp_order or model not in model_order:
            continue
        df = df[df[country_col].isin(top_countries)]
        df = df.set_index(country_col).reindex(top_countries).reset_index()
        data[rcp][model][year] = df["loss_as_gdp_pct"].fillna(0).values

    # --- Figure ---
    fig, ax = plt.subplots(
        figsize=(
            10,
            max(cfg["min_figure_height"], top_n * cfg["country_height_factor"]),
        ),
        dpi=cfg["figure"]["dpi"],
    )

    bar_width = cfg["bar_width"]
    y_pos = np.arange(len(top_countries)) * cfg["country_spacing"]

    rcp_cluster_gap = 4 * bar_width
    model_gap = bar_width

    rcp_offsets = {
        "RCP45": +rcp_cluster_gap / 2,
        "RCP85": -rcp_cluster_gap / 2,
    }
    model_offsets = {m: (i - 1) * model_gap for i, m in enumerate(model_order)}

    plt.rcParams["hatch.linewidth"] = cfg["hatch_linewidth"]

    # --- Draw bars ---
    for model in model_order:
        for rcp in rcp_order:
            y_vals = y_pos + rcp_offsets[rcp] + model_offsets[model]
            color = COLOURS[rcp.lower()]
            hatch = cfg["model_hatches"][model]

            for year, alpha in cfg["bar_alphas"].items():
                vals = data[rcp][model].get(year, np.zeros(len(top_countries)))
                ax.barh(
                    y_vals[::-1],
                    vals,
                    height=bar_width,
                    color=mcolors.to_rgba(color, alpha),
                    hatch=hatch,
                    edgecolor=cfg["bar_edgecolor"] if hatch else "none",
                    linewidth=cfg["bar_linewidth"] if hatch else 1,
                    zorder=1 if year == "2100" else 3,
                )

    # Create legends
    legend_cfg = cfg["legend"]

    # Scenario legend (RCP colors + year indicators)
    scenario_handles = [
        Patch(facecolor=get_scenario_color(rcp), edgecolor="none") for rcp in rcp_order
    ]
    scenario_handles.extend(
        [
            Patch(facecolor="grey", alpha=cfg["bar_alphas"]["2100"]),
            Patch(facecolor="grey", alpha=cfg["bar_alphas"]["2050"]),
        ]
    )
    scenario_labels = [RCP_LABELS[rcp] for rcp in rcp_order] + [
        "2100 (background)",
        "2050 (overlay)",
    ]

    # Model legend (hatches)
    model_handles = []
    for model in model_order:
        hatch = cfg["model_hatches"][model]
        if hatch:
            model_handles.append(
                Patch(facecolor="white", hatch=hatch, edgecolor="black")
            )
        else:
            model_handles.append(Patch(facecolor="grey", edgecolor="black"))

    legend1 = ax.legend(
        scenario_handles,
        scenario_labels,
        loc="upper right",
        fontsize=legend_cfg["fontsize"],
        title="Scenario",
        frameon=False,
        bbox_to_anchor=legend_cfg["scenario_bbox"],
        title_fontproperties={"weight": "bold", "size": legend_cfg["title_fontsize"]},
    )

    legend2 = ax.legend(
        model_handles,
        model_order,
        loc="upper right",
        fontsize=legend_cfg["fontsize"],
        title="Model",
        frameon=False,
        bbox_to_anchor=legend_cfg["model_bbox"],
        title_fontproperties={"weight": "bold", "size": legend_cfg["title_fontsize"]},
    )

    legend2._legend_box.align = "left"
    legend1._legend_box.align = "left"
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    # --- Country spine ---
    cluster_half_height = (
        abs(rcp_offsets["RCP45"] - rcp_offsets["RCP85"]) / 2
        + abs(model_offsets[model_order[-1]])
        + bar_width / 2
    )

    for y in y_pos[::-1]:
        ax.vlines(
            x=0,
            ymin=y - cluster_half_height,
            ymax=y + cluster_half_height,
            color=cfg["vline"]["color"],
            linewidth=cfg["vline"]["linewidth"],
            alpha=cfg["vline"]["alpha"],
            zorder=4,
        )

    # --- Axis formatting ---
    ax.set_yticks(y_pos[::-1])
    ax.set_yticklabels(
        [country_mapping.get(c, c).replace(" ", "\n") for c in top_countries],
        fontsize=cfg["axis"]["ylabel_fontsize"],
    )
    ax.set_xlabel(
        "Reef tourism loss as percentage of national GDP",
        fontsize=cfg["axis"]["xlabel_fontsize"],
    )

    ax.xaxis.set_minor_locator(plt.MultipleLocator(cfg["grid"]["minor_interval"]))
    ax.grid(
        True, axis="x", linestyle=":", alpha=cfg["grid"]["minor_alpha"], which="minor"
    )
    ax.grid(True, axis="x", linestyle=":", alpha=cfg["grid"]["major_alpha"])

    ax.tick_params(axis="y", length=0)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)

    # --- Legends (identical to scenario plot) ---
    # (reuse exactly the same legend code block)

    plt.tight_layout()
    plt.subplots_adjust(bottom=cfg["figure"]["bottom_margin"])

    if save_path:
        fig.savefig(save_path, dpi=cfg["figure"]["save_dpi"], bbox_inches="tight")

    return fig
