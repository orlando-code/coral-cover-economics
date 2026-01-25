from typing import Literal, Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.figure import Figure

from .plot_config import SpatialPlotConfig


def override_config_with_kwargs(config, **kwargs):
    """
    Override a config object with provided keyword arguments if they are not None.
    Returns a new config instance.
    """
    override_args = {}
    for key, value in kwargs.items():
        if key == "explode_factor":
            # Only override if not the default 1.0
            if value != 1.0:
                override_args[key] = value
        elif key in ["logarithmic_cbar", "show_scalebar", "scalebar_frame"]:
            # Only override if True
            if value:
                override_args[key] = value
        elif value is not None:
            override_args[key] = value
    if override_args:
        return config.update(**override_args)
    return config


def generate_geo_axis(
    figsize: tuple[float, float] = (10, 10),
    map_proj=None,
    dpi=300,
    central_longitude=None,
    central_latitude=None,
    config: Optional[SpatialPlotConfig] = None,
) -> tuple[Figure, Axes]:
    """Generate a geographical axes object.

    Args:
        figsize (tuple): the size of the figure.
        map_proj (ccrs.CRS): the projection of the map. If None, uses PlateCarree.
        dpi (int): the resolution of the figure.
        central_latitude (float, optional): the central latitude of the map, if supported.
        central_longitude (float, optional): the central longitude of the map, if supported.
        config (SpatialPlotConfig, optional): Configuration object. If provided, overrides
            individual parameters.

    Returns:
        tuple[Figure, Axes]: the figure and axes objects.
    """
    # Use config if provided
    if config is not None:
        figsize = config.figsize
        dpi = config.dpi
        central_longitude = config.central_longitude
        central_latitude = config.central_latitude
        map_proj = config.map_proj

    # Handle map_proj construction with central_longitude/latitude if provided
    if map_proj is None:
        # default to PlateCarree, may use central_longitude if given
        if central_longitude is not None:
            map_proj = ccrs.PlateCarree(central_longitude=central_longitude)
        else:
            map_proj = ccrs.PlateCarree()
    else:
        # Check if a central_longitude or central_latitude can/should be set
        proj_class = type(map_proj)
        proj_kwargs = {}
        if central_longitude is not None:
            if "central_longitude" in proj_class.__init__.__code__.co_varnames:
                proj_kwargs["central_longitude"] = central_longitude
        if central_latitude is not None:
            if "central_latitude" in proj_class.__init__.__code__.co_varnames:
                proj_kwargs["central_latitude"] = central_latitude
        # If kwargs for centering specified, try to reconstruct the projection
        if proj_kwargs:
            map_proj = proj_class(**proj_kwargs)
    return plt.figure(figsize=figsize, dpi=dpi), plt.axes(projection=map_proj)


def format_geo_axes(
    ax: plt.Axes,
    extent: Optional[tuple | list] = None,
    crs: Optional[ccrs.CRS] = None,
    landmass_zorder: int = 1000,
    ocean_zorder: int = -2,
    config: Optional[SpatialPlotConfig] = None,
) -> plt.Axes:
    """Format the geographical axes object.

    Args:
        ax (Axes): axes object to plot onto.
        extent (tuple, optional): the extent of the map (x0,x1,y0,y1).
        crs (ccrs.CRS, optional): the projection of the map.
        landmass_zorder (int): z-order for landmass features.
        config (SpatialPlotConfig, optional): Configuration object. If provided, overrides
            individual parameters.

    Returns:
        Axes: the formatted axes object.
    """
    # Use config if provided
    if config is not None:
        extent = config.get_extent() if config.extent is not None else extent
        crs = config.extent_crs if extent is not None else (crs or ccrs.PlateCarree())

        # Set extent if provided
        if extent is not None:
            ax.set_extent(extent, crs=crs)

        # Add map features based on config
        if config.show_land:
            ax.add_feature(
                cfeature.LAND, facecolor=config.land_color, zorder=landmass_zorder
            )
        if config.show_ocean:
            ax.add_feature(
                cfeature.OCEAN, alpha=config.ocean_alpha, zorder=ocean_zorder
            )
        if config.show_coastline:
            ax.add_feature(
                cfeature.COASTLINE,
                edgecolor=config.coastline_color,
                zorder=landmass_zorder,
            )
        if config.show_borders:
            ax.add_feature(
                cfeature.BORDERS,
                linestyle=config.border_linestyle,
                edgecolor=config.border_color,
                alpha=config.border_alpha,
                zorder=landmass_zorder,
            )
    else:
        # Default behavior
        if extent is None:
            extent = (-180, 180, -40, 50)
        if crs is None:
            crs = ccrs.PlateCarree()

    ax.set_extent(extent, crs=crs)
    ax.add_feature(cfeature.LAND, facecolor="white", zorder=landmass_zorder)
    ax.add_feature(cfeature.OCEAN, alpha=0.3, zorder=landmass_zorder)
    ax.add_feature(cfeature.COASTLINE, edgecolor="lightgray", zorder=landmass_zorder)
    ax.add_feature(
        cfeature.BORDERS,
        linestyle=":",
        edgecolor="gray",
        alpha=0.1,
        zorder=landmass_zorder,
    )

    return ax


COVARIATE_LABELS_DICT = {
    "lat_stzd": "Latitude",
    "historical_sst_max_stzd": "Max. historical SST",
    "ssta_dhwmax_stzd": "Max. SSTA DHW",
    "tsa_freqstdev_stzd": "TSA standard deviation",
    "ssta_min_stzd": "Min. SST anomaly",
    "tsa_max_stzd": "Max. TSA anomaly",
    "beta_diversity": "Beta diversity",
    "sst_mean_stzd": "Mean SST",
    "depth_stzd": "Depth",
    "ssta_mean_stzd": "Mean SSTA",
    "cyclone_stzd": "Cyclone frequency",
    "ssta_freqstdev_stzd": "SSTA frequency",
    "human_pop_stzd": "Local human population",
    "turbidity_mean_stzd": "Mean turbidity",
}


WA_COLORMAP = [
    "#3A9AB2",
    "#6FB2C1",
    "#91BAB6",
    "#A5C2A3",
    "#BDC881",
    "#DCCB4E",
    "#E3B710",
    "#E79805",
    "#EC7A05",
    "#EF5703",
    "#F11B00",
]


def get_wa_colormap(
    n_colours: int = 5, index: int = None, continuous: bool = False
) -> mcolors.ListedColormap | mcolors.LinearSegmentedColormap | str:
    """
    Get the Wes Anderson (Life Aquatic) colormap with n_colours equally spaced colors,
    or a specific color by index.

    Args:
        n_colours (int): The number of colours to include, equally spaced along
            the full colormap. Default is 5. Ignored if index is provided.
        index (int, optional): If provided, return a single color at this index
            from the full colormap. Index 0 is the first color, -1 is the last.
            Can use negative indexing.
        continuous (bool): If True, return a continuous LinearSegmentedColormap
            that smoothly interpolates between all colors. Default is False.

    Returns:
        ListedColormap, LinearSegmentedColormap, or str: If index is None and continuous=False,
            returns a ListedColormap with n_colours equally spaced colors.
            If index is None and continuous=True, returns a continuous LinearSegmentedColormap.
            If index is provided, returns a single color as a hex string.

    Examples:
        >>> cmap = get_wa_colormap(5)  # Get 5 equally spaced colors (discrete)
        >>> cmap = get_wa_colormap(256, continuous=True)  # Get continuous colormap
        >>> mid_color = get_wa_colormap(index=5)  # Get the 6th color (mid-point)
        >>> mid_color = get_wa_colormap(index=len(WA_COLORMAP)//2)  # True mid-point
        >>> last_color = get_wa_colormap(index=-1)  # Last color
    """
    if index is not None:
        # Return a specific color by index
        if abs(index) >= len(WA_COLORMAP):
            raise IndexError(
                f"Index {index} out of range. Colormap has {len(WA_COLORMAP)} colors "
                f"(valid indices: 0 to {len(WA_COLORMAP) - 1} or -{len(WA_COLORMAP)} to -1)"
            )
        return WA_COLORMAP[index]

    # Return colormap with n_colours equally spaced colors
    if n_colours <= 0:
        raise ValueError("n_colours must be positive")

    if continuous:
        # Return a continuous LinearSegmentedColormap that interpolates smoothly
        # between all colors in WA_COLORMAP
        return mcolors.LinearSegmentedColormap.from_list(
            "wa_colormap_continuous", WA_COLORMAP, N=n_colours
        )

    if n_colours >= len(WA_COLORMAP):
        # If requesting more or equal colors than available, return all
        selected_colors = WA_COLORMAP
    else:
        # Sample equally spaced indices
        indices = np.linspace(0, len(WA_COLORMAP) - 1, n_colours, dtype=int)
        selected_colors = [WA_COLORMAP[i] for i in indices]

    return mcolors.ListedColormap(selected_colors, name="wa_colormap")


# TODO: helper to get midpoint value


def transform_coordinates_for_central_longitude(gdf, central_longitude: float):
    """
    Transform GeoDataFrame coordinates to account for central_longitude shift.

    When using a map projection with central_longitude, the coordinate system
    is shifted. This function transforms the geometries to match.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with geometries to transform.
    central_longitude : float
        Central longitude of the map projection.

    Returns
    -------
    GeoDataFrame
        GeoDataFrame with transformed geometries.
    """

    def shift_longitude(geom, shift):
        """Shift longitude coordinates by shift degrees and wrap to [-180, 180]"""
        if geom.is_empty:
            return geom

        def transform_coords(coords):
            return [((lon + shift + 180) % 360 - 180, lat) for lon, lat in coords]

        from shapely.geometry import MultiPolygon, Polygon

        if geom.geom_type == "Polygon":
            exterior = transform_coords(geom.exterior.coords)
            interiors = [
                transform_coords(interior.coords) for interior in geom.interiors
            ]
            return Polygon(exterior, interiors)
        elif geom.geom_type == "MultiPolygon":
            return MultiPolygon(
                [
                    Polygon(
                        transform_coords(poly.exterior.coords),
                        [
                            transform_coords(interior.coords)
                            for interior in poly.interiors
                        ],
                    )
                    for poly in geom.geoms
                ]
            )
        else:
            return geom

    gdf_transformed = gdf.copy()
    shift = -central_longitude
    gdf_transformed["geometry"] = gdf_transformed.geometry.apply(
        lambda geom: shift_longitude(geom, shift)
    )

    return gdf_transformed


def _parse_location_string(loc: str) -> tuple[float, float, str, str]:
    """Parse location string into anchor point and alignment."""
    _loc_map = {
        "best": (0.5, 0.5, "center", "center"),
        "center": (0.5, 0.5, "center", "center"),
        "center left": (0.05, 0.5, "left", "center"),
        "left": (0.05, 0.5, "left", "center"),
        "center right": (0.95, 0.5, "right", "center"),
        "right": (0.95, 0.5, "right", "center"),
        "lower center": (0.5, 0.07, "center", "bottom"),
        "upper center": (0.5, 0.93, "center", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "leftmiddle": (0.05, 0.5, "left", "center"),
        "lefttop": (0.05, 0.93, "left", "top"),
        "leftbottom": (0.05, 0.07, "left", "bottom"),
        "upper left": (0.05, 0.93, "left", "top"),
        "upper right": (0.95, 0.93, "right", "top"),
        "lower right": (0.95, 0.02, "right", "bottom"),
    }
    loc_aliases = {
        "leftbottom": "lower left",
        "lefttop": "upper left",
        "leftmiddle": "center left",
        "rightbottom": "lower right",
        "righttop": "upper right",
        "rightmiddle": "center right",
        "centermiddle": "center",
    }

    loc = loc.lower().replace("_", " ")
    if loc in loc_aliases:
        loc = loc_aliases[loc]
    if loc not in _loc_map:
        loc = "lower left"
    return _loc_map[loc]


def _calculate_scalebar_length(extent: tuple, units: str) -> float:
    """Calculate appropriate scalebar length based on map extent."""
    x0, x1, y0, y1 = extent
    deg_length = (x1 - x0) * 0.5

    if units == "degrees":
        length = deg_length
        if length >= 10:
            length = round(length / 10) * 10
        elif length >= 5:
            length = round(length / 5) * 5
        elif length >= 1:
            length = round(length)
        else:
            length = round(length * 2) / 2
    elif units == "km":
        km_length = deg_length * 111.0
        if km_length >= 1000:
            length = round(km_length / 1000) * 1000
        elif km_length >= 500:
            length = round(km_length / 500) * 500
        elif km_length >= 100:
            length = round(km_length / 100) * 100
        elif km_length >= 50:
            length = round(km_length / 50) * 50
        else:
            length = round(km_length / 10) * 10
    else:  # miles
        miles_length = deg_length * 69.0
        if miles_length >= 500:
            length = round(miles_length / 500) * 500
        elif miles_length >= 100:
            length = round(miles_length / 100) * 100
        elif miles_length >= 50:
            length = round(miles_length / 50) * 50
        else:
            length = round(miles_length / 10) * 10

    return length


def _axis_text_height(ax: plt.Axes, fontsize: int) -> float:
    """Estimate text height in axis coordinates."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    text_obj = ax.text(0, 0, "0", fontsize=fontsize, transform=ax.transAxes)
    bbox = text_obj.get_window_extent(renderer=renderer)
    ax_bbox = ax.get_window_extent()
    text_obj.remove()
    return bbox.height / ax_bbox.height if ax_bbox.height > 0 else 0.04


def _estimate_text_width_axis(ax: plt.Axes, text: str, fontsize: int) -> float:
    """Estimate text width in axis coordinates."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    text_obj = ax.text(0, 0, text, fontsize=fontsize, transform=ax.transAxes)
    bbox = text_obj.get_window_extent(renderer=renderer)
    ax_bbox = ax.get_window_extent()
    text_obj.remove()
    return bbox.width / ax_bbox.width if ax_bbox.width > 0 else 0.05


def _format_tick_label(value: float, units: str, tick_increment: float) -> str:
    """Format tick label based on value and units."""
    if units == "degrees":
        if tick_increment >= 1:
            return f"{int(value)}"
        elif tick_increment >= 0.5:
            return f"{value:.1f}"
        else:
            return f"{value:.2f}"
    elif units == "km":
        if value >= 1000:
            return f"{int(value / 1000)}k"
        else:
            return f"{int(value)}"
    else:  # miles
        return f"{int(value)}"


def _calculate_vertical_positions(
    anchor_y: float,
    valign: str,
    bar_height_axis: float,
    tick_height_axis: float,
    label_spacing_axis: float,
    units_spacing_axis: float,
    ticklabel_height_axis: float,
    ticklabel_vertical_gap: float,
    fontsize: int,
    bbox_height: float,
) -> dict[str, float]:
    """Calculate vertical positions for bar, ticks, labels, and units."""
    units_text_height = (fontsize / 72.0) / bbox_height

    if valign == "bottom":
        bar_y = (
            anchor_y
            + bar_height_axis / 2
            + ticklabel_height_axis
            + tick_height_axis
            + label_spacing_axis
            + ticklabel_vertical_gap
        )
        units_y = bar_y + units_spacing_axis
        label_y = bar_y - tick_height_axis - label_spacing_axis
        tick_bottom = bar_y - tick_height_axis
    elif valign == "top":
        bar_y = anchor_y - bar_height_axis / 2 - units_spacing_axis - units_text_height
        units_y = bar_y + units_spacing_axis + bar_height_axis / 2 + units_text_height
        label_y = bar_y - tick_height_axis - label_spacing_axis
        tick_bottom = bar_y - tick_height_axis
    else:  # center
        bar_y = anchor_y
        units_y = bar_y + units_spacing_axis
        label_y = bar_y - tick_height_axis - label_spacing_axis
        tick_bottom = bar_y - tick_height_axis

    print(bar_y, bar_y - bar_height_axis / 2, bar_y + bar_height_axis / 2, tick_bottom)
    return {
        "bar_y": bar_y,
        "bar_bottom": bar_y - bar_height_axis / 2,
        "bar_top": bar_y + bar_height_axis / 2,
        "units_y": units_y,
        "label_y": label_y,
        "tick_bottom": tick_bottom,
    }


def _calculate_horizontal_position(
    anchor_x: float, halign: str, length_axis: float
) -> float:
    """Calculate horizontal position of scalebar based on alignment."""
    if halign == "left":
        return anchor_x
    elif halign == "right":
        return anchor_x - length_axis
    else:  # center
        return anchor_x - length_axis / 2


def _draw_scalebar_segments(
    ax: plt.Axes,
    x_pos: float,
    bar_y: float,
    bar_height: float,
    segment_length: float,
    segments: int,
    color: str,
    zorder: int,
) -> list[float]:
    """Draw scalebar segments and return tick positions."""
    from matplotlib.patches import Rectangle

    tick_positions = [x_pos]
    for i in range(segments):
        segment_color = color if i % 2 == 0 else "white"
        x_start = x_pos + i * segment_length
        x_end = x_pos + (i + 1) * segment_length

        ax.add_patch(
            Rectangle(
                (x_start, bar_y - bar_height / 2),
                x_end - x_start,
                bar_height,
                facecolor=segment_color,
                edgecolor=None,
                transform=ax.transAxes,
                zorder=zorder,
                clip_on=False,
            )
        )
        tick_positions.append(x_end)

    return tick_positions


def _draw_ticks(
    ax: plt.Axes,
    tick_positions: list[float],
    bar_bottom: float,
    tick_bottom: float,
    linewidth: float,
    zorder: int,
):
    """Draw tick marks."""
    for tick_x in tick_positions:
        ax.plot(
            [tick_x, tick_x],
            [bar_bottom, tick_bottom],
            color="black",
            linewidth=linewidth * 0.2,
            transform=ax.transAxes,
            zorder=zorder - 1,
            clip_on=False,
        )


def _draw_tick_labels(
    ax: plt.Axes,
    tick_positions: list[float],
    tick_increment: float,
    label_y: float,
    units: str,
    fontsize: int,
    color: str,
    tick_rotation: float,
    zorder: int,
):
    """Draw tick value labels."""
    for i, tick_x in enumerate(tick_positions):
        tick_value = i * tick_increment
        label_text = _format_tick_label(tick_value, units, tick_increment)

        ax.text(
            tick_x,
            label_y,
            label_text,
            ha="center",
            va="top",
            fontsize=fontsize * 0.9,
            color=color,
            transform=ax.transAxes,
            zorder=zorder,
            clip_on=False,
            rotation=tick_rotation,
        )


def _draw_units_label(
    ax: plt.Axes,
    units_x: float,
    units_y: float,
    units: str,
    fontsize: int,
    color: str,
    zorder: int,
):
    """Draw units label above scalebar."""
    units_text = units.upper() if units in ["km", "miles"] else "Degrees"
    ax.text(
        units_x,
        units_y,
        units_text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        transform=ax.transAxes,
        zorder=zorder,
        clip_on=False,
    )


def _draw_frame(
    ax: plt.Axes,
    x_pos: float,
    length_axis: float,
    positions: dict[str, float],
    halign: str,
    valign: str,
    units: str,
    tick_increment: float,
    length: float,
    fontsize: int,
    horizontal_pad: float,
    vertical_pad: float,
    zorder: int,
):
    """Draw frame around scalebar."""
    from matplotlib.patches import Rectangle

    # Calculate longest tick label for width estimation
    max_tick_value = length
    longest_label = _format_tick_label(max_tick_value, units, tick_increment)
    tick_label_width = _estimate_text_width_axis(ax, longest_label, fontsize * 0.9)

    # Horizontal bounds
    bar_left = x_pos
    bar_right = x_pos + length_axis
    label_extension = max(tick_label_width / 2, horizontal_pad)

    if halign == "left":
        # frame_left = 0.0
        frame_left = bar_left - label_extension
        frame_right = bar_right + label_extension
    elif halign == "right":
        frame_left = bar_left - label_extension
        # frame_right = 1.0
        frame_right = bar_right + label_extension
    else:  # center
        frame_left = bar_left - label_extension
        frame_right = bar_right + label_extension

    # Vertical bounds - calculate symmetrically around bar center
    tick_label_height = _axis_text_height(ax, fontsize * 1)
    units_text_height = _axis_text_height(ax, fontsize)

    bar_center = positions["bar_y"]

    # Calculate distances from bar center to top and bottom content edges
    content_top_edge = max(
        positions["units_y"] + units_text_height,
        positions["bar_top"],
    )
    content_bottom_edge = min(
        positions["label_y"] - tick_label_height,
        positions["tick_bottom"],
        positions["bar_bottom"],
    )

    # Calculate distances from bar center
    distance_above = content_top_edge - bar_center
    distance_below = bar_center - content_bottom_edge
    print(distance_above, distance_below)

    # Use the larger distance for symmetry, then add padding
    max_distance = max(distance_above, distance_below)
    frame_top = bar_center + max_distance + vertical_pad
    frame_bottom = bar_center - max_distance - vertical_pad
    print(frame_top, frame_bottom)

    ax.add_patch(
        Rectangle(
            (frame_left, frame_bottom),
            frame_right - frame_left,
            frame_top - frame_bottom,
            fill=True,
            facecolor="white",
            alpha=0.8,
            linewidth=0.8,
            transform=ax.transAxes,
            zorder=zorder - 2,
            clip_on=False,
        )
    )


def add_scalebar(
    ax: plt.Axes,
    length: float = None,
    location: tuple[float, float] = None,
    units: Literal["degrees", "km", "miles"] = "degrees",
    segments: int = 4,
    linewidth: float = 2.0,
    fontsize: int = 10,
    color: str = "black",
    zorder: int = 1000,
    tick_rotation: float = 30,
    frame: bool = False,
    loc: Literal[
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
    ] = "lower right",
) -> None:
    """
    Add a classic black and white dashed scalebar to a cartopy map.

    Parameters
    ----------
    ax : Axes
        Cartopy axes to add scalebar to.
    length : float, optional
        Length of scalebar in the specified units. If None, auto-calculates based on extent.
    location : tuple[float, float], optional
        Location of scalebar in axes coordinates (xmin, ymin). If None, determined by `loc`.
    units : str, optional
        Units for scalebar: "degrees", "km", or "miles". Default: "degrees".
    segments : int, optional
        Number of alternating black/white segments. Default: 4.
    linewidth : float, optional
        Width of scalebar line. Default: 2.0.
    fontsize : int, optional
        Font size for scalebar label. Default: 10.
    color : str, optional
        Color for scalebar (black segments). Default: "black".
    zorder : int, optional
        Z-order for scalebar. Default: 1000.
    tick_rotation : float, optional
        Rotation of tick labels. Default: 0.
    frame : bool, optional
        Whether to draw a frame (white transluscent back, black edge) around the scalebar. Default: False.
    loc : str, optional
        Legend location string (e.g. "lower left", "center", "upper right", etc.) using matplotlib legend/naming convention.
        If `location` is given, that takes precedence.
    Returns
    -------
    None
    """
    import cartopy.crs as ccrs

    # Parse location string
    anchor_x, anchor_y, halign, valign = _parse_location_string(loc)
    # Override with explicit location if provided
    if location is not None:
        anchor_x, anchor_y = location
        halign = "center"
        valign = "center"

    # Get map extent and calculate length
    extent = ax.get_extent(ccrs.PlateCarree())
    x0, x1, y0, y1 = extent
    if length is None:
        length = _calculate_scalebar_length(extent, units)

    # Convert to axis coordinates
    x_range_data = x1 - x0
    length_axis = length / x_range_data

    # Calculate dimensions in axis coordinates
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bar_height_axis = (linewidth / 72.0) / bbox.height

    # Spacing constants
    tick_height_axis = 0.015
    label_spacing_axis = 0.01
    units_spacing_axis = 0.01
    horizontal_pad = 0.01
    vertical_pad = 0.01

    # Estimate tick label height
    try:
        ticklabel_fontsize = max(
            [label.get_size() for label in ax.get_xticklabels() if label.get_text()]
        )
        ticklabel_height_axis = _axis_text_height(ax, ticklabel_fontsize)
    except Exception:
        ticklabel_height_axis = 0.04
    ticklabel_vertical_gap = 0.012

    # Calculate positions
    positions = _calculate_vertical_positions(
        anchor_y,
        valign,
        bar_height_axis,
        tick_height_axis,
        label_spacing_axis,
        units_spacing_axis,
        ticklabel_height_axis,
        ticklabel_vertical_gap,
        fontsize,
        bbox.height,
    )

    x_pos_axis = _calculate_horizontal_position(anchor_x, halign, length_axis)
    segment_length_axis = length_axis / segments
    tick_increment = length / segments

    # Add bar positions to dict for frame calculation
    positions["bar_left"] = x_pos_axis
    positions["bar_right"] = x_pos_axis + length_axis

    # Draw scalebar elements
    tick_positions = _draw_scalebar_segments(
        ax,
        x_pos_axis,
        positions["bar_y"],
        bar_height_axis,
        segment_length_axis,
        segments,
        color,
        zorder,
    )

    _draw_ticks(
        ax,
        tick_positions,
        positions["bar_bottom"],
        positions["tick_bottom"],
        linewidth,
        zorder,
    )

    _draw_tick_labels(
        ax,
        tick_positions,
        tick_increment,
        positions["label_y"],
        units,
        fontsize,
        color,
        tick_rotation,
        zorder,
    )

    _draw_units_label(
        ax,
        x_pos_axis + length_axis / 2,
        positions["units_y"],
        units,
        fontsize,
        color,
        zorder,
    )

    # Draw frame if requested
    if frame:
        _draw_frame(
            ax,
            x_pos_axis,
            length_axis,
            positions,
            halign,
            valign,
            units,
            tick_increment,
            length,
            fontsize,
            horizontal_pad,
            vertical_pad,
            zorder,
        )


class ThresholdLogNorm(Normalize):
    """
    Normalization that maps values < threshold to grey, then logarithmic.

    Values below the threshold are mapped to a grey portion of the colormap,
    and values >= threshold are mapped logarithmically to the colored portion.
    """

    def __init__(self, threshold, vmin, vmax, clip=False, grey_fraction=0.1):
        """
        Parameters
        ----------
        threshold : float
            Value below which values are shown as grey.
        vmin : float
            Minimum value for normalization.
        vmax : float
            Maximum value for normalization.
        clip : bool, optional
            Whether to clip values to [vmin, vmax]. Default: False.
        grey_fraction : float, optional
            Fraction of colorbar to use for grey portion. Default: 0.1 (10%).
        """
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.threshold = threshold
        self.grey_fraction = grey_fraction
        # Grey portion is [0, grey_fraction], color portion is [grey_fraction, 1]
        self.threshold_norm = grey_fraction
        self.log_norm = LogNorm(vmin=threshold, vmax=vmax, clip=clip)

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        value = np.asarray(value)
        result = np.zeros_like(value, dtype=float)
        below_threshold = value < self.threshold
        above_threshold = ~below_threshold

        # Values below threshold: map linearly to [0, threshold_norm]
        if np.any(below_threshold):
            threshold_range = self.threshold - self.vmin
            if threshold_range > 0:
                normalized = (value[below_threshold] - self.vmin) / threshold_range
            else:
                normalized = np.zeros_like(value[below_threshold])
            normalized = np.clip(normalized, 0, 1) if clip else normalized
            result[below_threshold] = normalized * self.threshold_norm

        # Values >= threshold: map logarithmically to [threshold_norm, 1]
        if np.any(above_threshold):
            log_normalized = self.log_norm(value[above_threshold], clip=clip)
            # Map from [0, 1] to [threshold_norm, 1]
            result[above_threshold] = self.threshold_norm + log_normalized * (
                1 - self.threshold_norm
            )

        return result

    def inverse(self, value):
        """Inverse mapping for colorbar."""
        value = np.asarray(value)
        result = np.zeros_like(value, dtype=float)
        below_threshold_norm = value < self.threshold_norm
        above_threshold_norm = ~below_threshold_norm

        if np.any(below_threshold_norm):
            # Linear inverse for grey portion
            normalized = value[below_threshold_norm] / self.threshold_norm
            threshold_range = self.threshold - self.vmin
            result[below_threshold_norm] = self.vmin + normalized * threshold_range

        if np.any(above_threshold_norm):
            # Logarithmic inverse for color portion
            # Map from [threshold_norm, 1] back to [0, 1] for log_norm
            normalized = (value[above_threshold_norm] - self.threshold_norm) / (
                1 - self.threshold_norm
            )
            # Ensure normalized values are in [0, 1]
            normalized = np.clip(normalized, 0, 1)
            result[above_threshold_norm] = self.log_norm.inverse(normalized)

        return result

    def autoscale_None(self, A):
        """Autoscale to handle None vmin/vmax."""
        # This is needed for matplotlib compatibility
        if self.vmin is None or self.vmax is None:
            A = np.asarray(A)
            if self.vmin is None:
                self.vmin = A.min()
            if self.vmax is None:
                self.vmax = A.max()
            # Update log_norm with new vmax
            self.log_norm = LogNorm(vmin=self.threshold, vmax=self.vmax, clip=self.clip)


def create_threshold_log_colormap(base_cmap, threshold=10.0, grey_fraction=0.1):
    """
    Create a colormap with grey for values < threshold, then logarithmic colors.

    Parameters
    ----------
    base_cmap : matplotlib.colors.Colormap
        Base colormap to use for values >= threshold.
    threshold : float, optional
        Threshold value. Default: 10.0.
    grey_fraction : float, optional
        Fraction of colorbar to use for grey portion. Default: 0.1 (10%).

    Returns
    -------
    matplotlib.colors.ListedColormap
        Combined colormap with grey and colored portions.
    """
    grey_color = np.array([0.7, 0.7, 0.7, 1.0])  # Light grey

    # Calculate number of colors for each portion
    # Grey portion should be grey_fraction of total
    # If we want n_color colors in the colored portion, then:
    # n_grey / (n_grey + n_color) = grey_fraction
    # n_grey = grey_fraction * (n_grey + n_color)
    # n_grey = grey_fraction * n_grey + grey_fraction * n_color
    # n_grey * (1 - grey_fraction) = grey_fraction * n_color
    # n_grey = (grey_fraction * n_color) / (1 - grey_fraction)

    n_color = 256  # Number of colors in the colored portion
    n_grey = int((grey_fraction * n_color) / (1 - grey_fraction))

    # Create grey portion
    grey_colors = np.tile(grey_color, (n_grey, 1))

    # Get colors from base colormap for values >= threshold
    color_values = np.linspace(0, 1, n_color)
    color_colors = base_cmap(color_values)

    # Combine: grey first, then colors
    combined_colors = np.vstack([grey_colors, color_colors])
    return ListedColormap(combined_colors)


def limit_line_length(line: str, line_lim: int = 20) -> str:
    """If string is longer than line_lim, insert a newline at nearest-to-over whitespace until there are no non-broken strings which are longer than line_lim"""
    line_len = len(line)

    if line_len < line_lim:
        return line

    # find the nearest whitespace to the line_lim
    whitespace_idx = line[:line_lim].rfind(" ")
    if whitespace_idx == -1:
        return line

    # insert a newline at the whitespace
    return (
        line[:whitespace_idx]
        + "\n"
        + limit_line_length(line[whitespace_idx + 1 :], line_lim)
    )
