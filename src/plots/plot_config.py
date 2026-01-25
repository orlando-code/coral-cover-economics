"""
Plotting configuration system for reproducible, flexible figure generation.

This module provides configuration classes for consistent plot formatting,
especially useful for paper figures where consistent styling is critical.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import cartopy.crs as ccrs


@dataclass
class PlotConfig:
    """Base configuration for all plots."""

    # Figure settings
    figsize: Tuple[float, float] = (12, 8)
    dpi: int = 300

    # Font settings
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10

    # Color settings
    cmap: str = "turbo"
    edgecolor: Optional[str] = None
    edgewidth: float = 0.2

    # Colorbar settings
    cbar_orientation: Literal["horizontal", "vertical"] = "horizontal"
    cbar_pad: float = 0.1
    cbar_aspect: float = 50
    cbar_label_fontsize: int = 11

    # Title and labels
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None

    # Save settings
    save_dpi: int = 300
    save_format: str = "png"
    tight_layout: bool = True

    def copy(self) -> "PlotConfig":
        """Create a copy of this config."""
        return PlotConfig(**self.__dict__)

    def update(self, **kwargs) -> "PlotConfig":
        """Update config with new values and return a new instance."""
        new_dict = self.__dict__.copy()
        new_dict.update(kwargs)
        return PlotConfig(**new_dict)


@dataclass
class SpatialPlotConfig(PlotConfig):
    """Configuration for spatial plots with map projections."""

    label_fontsize: int = 12
    title_fontsize: int = 14
    tick_fontsize: int = 10
    save_dpi: int = 300
    save_format: str = "png"
    tight_layout: bool = True

    # Map projection settings
    central_longitude: Optional[float] = None
    central_latitude: Optional[float] = None
    map_proj: Optional[ccrs.CRS] = None

    # Extent settings (x0, x1, y0, y1) in degrees
    extent: Optional[Tuple[float, float, float, float]] = None
    extent_crs: ccrs.CRS = field(default_factory=lambda: ccrs.PlateCarree())

    # Map features
    show_land: bool = True
    show_ocean: bool = True
    show_coastline: bool = True
    show_borders: bool = True
    land_color: str = "white"
    ocean_alpha: float = 0.3
    coastline_color: str = "lightgray"
    border_color: str = "gray"
    border_alpha: float = 0.1
    border_linestyle: str = ":"

    # Data transformation
    explode_factor: float = 1.0

    # Color scale settings
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    logarithmic_cbar: bool = False
    cmap_name: str = (
        "turbo"  # Can be "turbo", "life aquatic", or any matplotlib colormap name
    )
    cbar_pad: float = 0.1
    cbar_title: Optional[str] = None
    cbar_shrink: float = 1.0
    # Coordinate transformation for central_longitude
    transform_coords: bool = (
        True  # Whether to transform coords when central_longitude is set
    )

    # Scalebar settings
    show_scalebar: bool = True
    scalebar_length: Optional[float] = None  # Auto-calculate if None
    scalebar_location: Optional[Tuple[float, float]] = (
        None  # Axes coordinates (x, y). If None, uses scalebar_loc
    )
    scalebar_loc: Literal[
        "best",
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
    ] = "lower right"  # Location string (matplotlib legend convention)
    scalebar_units: Literal["degrees", "km", "miles"] = "degrees"
    scalebar_segments: int = 4
    scalebar_linewidth: float = 5
    scalebar_fontsize: int = 10
    scalebar_color: str = "black"
    scalebar_tick_rotation: float = 0
    scalebar_frame: bool = False

    def get_projection(self) -> ccrs.CRS:
        """Get the cartopy projection based on config."""
        if self.map_proj is not None:
            return self.map_proj

        if self.central_longitude is not None:
            return ccrs.PlateCarree(central_longitude=self.central_longitude)

        return ccrs.PlateCarree()

    def get_extent(self) -> Optional[Tuple[float, float, float, float]]:
        """Get the extent, transforming if needed for central_longitude."""
        if self.extent is None:
            return None

        # If central_longitude is set and we need to transform, do it
        if self.central_longitude is not None and self.transform_coords:
            x0, x1, y0, y1 = self.extent
            # Transform longitude coordinates
            x0_transformed = ((x0 - self.central_longitude + 180) % 360) - 180
            x1_transformed = ((x1 - self.central_longitude + 180) % 360) - 180
            return (x0_transformed, x1_transformed, y0, y1)

        return self.extent

    def get_colormap(self):
        """
        Get the colormap based on cmap_name or cmap field.

        If cmap_name is "life aquatic", returns the continuous Wes Anderson colormap.
        Otherwise, returns the standard matplotlib colormap name.

        Returns:
            str or LinearSegmentedColormap: Colormap name (str) or continuous Wes Anderson colormap
        """
        from src.plots import plot_utils

        # Check cmap_name first (SpatialPlotConfig specific)
        if hasattr(self, "cmap_name") and self.cmap_name.lower() == "life aquatic":
            # Return continuous Wes Anderson colormap with 256 colors
            return plot_utils.get_wa_colormap(n_colours=256, continuous=True)

        # Check cmap field (from PlotConfig base class)
        if self.cmap.lower() == "life aquatic":
            return plot_utils.get_wa_colormap(n_colours=256, continuous=True)

        # Use cmap_name if available, otherwise fall back to cmap
        return getattr(self, "cmap_name", self.cmap)

    def copy(self) -> "SpatialPlotConfig":
        """Create a copy of this config."""
        return SpatialPlotConfig(**self.__dict__)

    def update(self, **kwargs) -> "SpatialPlotConfig":
        """Update config with new values and return a new instance."""
        new_dict = self.__dict__.copy()
        new_dict.update(kwargs)
        return SpatialPlotConfig(**new_dict)


# Predefined configs for common use cases
PAPER_CONFIG = PlotConfig(
    figsize=(10, 8),
    dpi=300,
    title_fontsize=14,
    label_fontsize=12,
    tick_fontsize=10,
    save_dpi=300,
)

PAPER_SPATIAL_CONFIG = SpatialPlotConfig(
    figsize=(12, 8),
    dpi=300,
    title_fontsize=14,
    label_fontsize=12,
    tick_fontsize=10,
    save_dpi=300,
    show_land=True,
    show_ocean=True,
    show_coastline=True,
    show_borders=True,
)

# Global extent presets (x0, x1, y0, y1) in degrees
EXTENT_PRESETS = {
    "global": (-180, 180, -90, 90),
    "global_reef": (-180, 180, -40, 40),
    "pacific": (100, 260, -40, 40),
    "atlantic": (-80, 20, -40, 50),
    "indian": (20, 150, -40, 30),
    "caribbean": (-85, -60, 10, 28),
    "indo_pacific": (30, 300, -35, 35),  # 30E to 60W
    "europe": (-10, 50, 35, 70),
    "asia_pacific": (100, 180, -10, 50),
    "florida": (-85, -60, 10, 30),
    "florida_focus": (-85, -70, 20, 27),
    "florida_focus_zoom": (-83.5, -80, 24, 25),
    "red_sea": (32, 42, 20, 30),
}


MODEL_COLORS = {
    "linear": "#E3B710",
    "compound": "#3A9AB2",
    "tipping_point": "#F11B00",
}


# countries mapping
