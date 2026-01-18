import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
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
