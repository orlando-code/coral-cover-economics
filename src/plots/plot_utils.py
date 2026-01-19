import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def generate_geo_axis(
    figsize: tuple[float, float] = (10, 10),
    map_proj=None,
    dpi=300,
    central_longitude=None,
    central_latitude=None,
) -> tuple[Figure, Axes]:
    """Generate a geographical axes object.

    Args:
        figsize (tuple): the size of the figure.
        map_proj (ccrs.CRS): the projection of the map. If None, uses PlateCarree.
        dpi (int): the resolution of the figure.
        central_latitude (float, optional): the central latitude of the map, if supported.
        central_longitude (float, optional): the central longitude of the map, if supported.

    Returns:
        tuple[Figure, Axes]: the figure and axes objects.
    """
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
    extent: tuple | list = (-180, 180, -40, 50),
    crs: ccrs.CRS = ccrs.PlateCarree(),
    landmass_zorder: int = -1,
) -> plt.Axes:
    """Format the geographical axes object.

    Args:
        ax (Axes): axes object to plot onto.
        extent (tuple): the extent of the map.
        crs (ccrs.CRS): the projection of the map.

    Returns:
        Axes: the formatted axes object.
    """
    ax.set_extent(extent, crs=crs)
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.add_feature(cfeature.OCEAN, alpha=0.3)
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
    n_colours: int = 5, index: int = None
) -> mcolors.ListedColormap | str:
    """
    Get the Wes Anderson (Life Aquatic) colormap with n_colours equally spaced colors,
    or a specific color by index.

    Args:
        n_colours (int): The number of colours to include, equally spaced along
            the full colormap. Default is 5. Ignored if index is provided.
        index (int, optional): If provided, return a single color at this index
            from the full colormap. Index 0 is the first color, -1 is the last.
            Can use negative indexing.

    Returns:
        ListedColormap or str: If index is None, returns a ListedColormap with
            n_colours equally spaced colors. If index is provided, returns a
            single color as a hex string.

    Examples:
        >>> cmap = get_wa_colormap(5)  # Get 5 equally spaced colors
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

    if n_colours >= len(WA_COLORMAP):
        # If requesting more or equal colors than available, return all
        selected_colors = WA_COLORMAP
    else:
        # Sample equally spaced indices
        indices = np.linspace(0, len(WA_COLORMAP) - 1, n_colours, dtype=int)
        selected_colors = [WA_COLORMAP[i] for i in indices]

    return mcolors.ListedColormap(selected_colors, name="wa_colormap")


# TODO: helper to get midpoint value
