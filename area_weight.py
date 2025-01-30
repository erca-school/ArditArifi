import numpy as np
import xarray as xr
from haversine import haversine, Unit

def area_weight_haversine(lat_values, lon_values):
    """
    Compute grid-cell area weights (normalized) using the haversine formula.
    
    Parameters
    ----------
    lat_values : numpy.ndarray
        1D array of latitude values (degrees).
    lon_values : numpy.ndarray
        1D array of longitude values (degrees).
        
    Returns
    -------
    weights : 2D numpy.ndarray
        2D array (lat x lon) of normalized area weights.
    """
    lats = lat_values
    lons = lon_values

    
    surf = np.zeros((len(lats),len(lons)))
    lons_ = np.copy(lons)
    lons_[lons_>180] = lons_[lons_>180]-360.0
    for k in range(len(lats)-1):
        for l in range(len(lons)-1):
            p1 = (lats[k],lons_[k])
            p2 = (lats[k],lons_[k+1])
            dx = haversine(p1,p2,unit=Unit.METERS)
            p1 = (lats[k],lons_[k])
            p2 = (lats[k+1],lons_[k])
            dy = haversine(p1,p2,unit=Unit.METERS)
            surf[k,l] = dx*dy
    surf = surf/np.max(surf)
    if np.max(lats)<0:
        surf[-1,:]=1
    if np.min(lats)>0:
        surf[-1,:]=np.min(surf[surf>0])
        
    return surf


def area_weight_haversine_2(lat_values, lon_values):
    """
    Compute grid-cell area weights (normalized) using the haversine formula but without normalizing before.
    
    Parameters
    ----------
    lat_values : numpy.ndarray
        1D array of latitude values (degrees).
    lon_values : numpy.ndarray
        1D array of longitude values (degrees).
        
    Returns
    -------
    weights : 2D numpy.ndarray
        2D array (lat x lon) of normalized area weights.
    """
    lats = lat_values
    lons = lon_values

    
    surf = np.zeros((len(lats),len(lons)))
    lons_ = np.copy(lons)
    lons_[lons_>180] = lons_[lons_>180]-360.0
    for k in range(len(lats)-1):
        for l in range(len(lons)-1):
            p1 = (lats[k],lons_[k])
            p2 = (lats[k],lons_[k+1])
            dx = haversine(p1,p2,unit=Unit.METERS)
            p1 = (lats[k],lons_[k])
            p2 = (lats[k+1],lons_[k])
            dy = haversine(p1,p2,unit=Unit.METERS)
            surf[k,l] = dx*dy
    if np.max(lats)<0:
        surf[-1,:]=1
    if np.min(lats)>0:
        surf[-1,:]=np.min(surf[surf>0])
        
    return surf
    
def area_weight_cosine(lat_values, lon_values):
    """
    Compute approximate area weights proportional to cos(latitude).
    Useful for many regular lat-lon grids.
    
    Parameters
    ----------
    lat_values : numpy.ndarray
        1D array of latitude values (degrees).
    lon_values : numpy.ndarray
        1D array of longitude values (degrees).
        
    Returns
    -------
    weights : 2D numpy.ndarray
        2D array (lat x lon) of area weights.
        Not normalized here; it’s just cos(lat).
    """
    # Convert lat to radians
    lat_radians = np.deg2rad(lat_values)
    # cos(lat) for each lat
    cos_lat = np.cos(lat_radians)
    # Create 2D weights by broadcasting
    # shape will be (len(lat), len(lon))
    weights_2d = cos_lat[:, np.newaxis] * np.ones((1, len(lon_values)))
    return weights_2d

import numpy as np

def area_weight_r_sinlat_regular(lat_values, lon_values, radius=6371000.0):
    """
    Compute approximate grid-cell areas (or area weights) for a regular
    lat-lon grid, assuming lat_values and lon_values are midpoints (cell centers).
    We use:
        area = r^2 * [sin(lat+Δlat/2) - sin(lat−Δlat/2)] * Δlon
    for each cell, where r is Earth's radius in meters.

    Parameters
    ----------
    lat_values : np.ndarray
        1D array of lat midpoints (degrees), uniformly spaced.
        Must have at least 2 elements (ascending order).
    lon_values : np.ndarray
        1D array of lon midpoints (degrees), uniformly spaced.
        Must have at least 2 elements (ascending order).
    radius : float, optional
        Earth radius in meters (default=6371000.0).

    Returns
    -------
    area_2d : np.ndarray
        2D array of shape (n_lat, n_lon) with approximate cell areas in m^2.

    Notes
    -----
    - This function is valid for a regularly spaced grid in both lat & lon.

    """
    # Convert lat, lon midpoints to radians
    lat_rad = np.deg2rad(lat_values)
    lon_rad = np.deg2rad(lon_values)

    nlat = len(lat_values)
    nlon = len(lon_values)

    # Calculate uniform spacing in lat/lon (in degrees -> convert to radians)
    dlat_deg = lat_values[1] - lat_values[0]
    dlat_rad = np.deg2rad(dlat_deg)

    dlon_deg = lon_values[1] - lon_values[0]
    dlon_rad = np.deg2rad(dlon_deg)

    # Prepare the output array
    area_2d = np.zeros((nlat, nlon), dtype=float)

    # For each cell [i, j]:
    # lat[i] ± dlat/2, and a uniform Δlon = dlon_rad
    for i in range(nlat):
        # Northern/southern edges for this cell
        lat_north = lat_rad[i] + 0.5 * dlat_rad
        lat_south = lat_rad[i] - 0.5 * dlat_rad

        # Δ(sin(lat)) = sin(lat_north) - sin(lat_south)
        sin_term = np.sin(lat_north) - np.sin(lat_south)

        for j in range(nlon):
            # Because spacing is uniform, each cell has Δlon = dlon_rad
            # area = R^2 * [sin(lat_north) - sin(lat_south)] * Δlon
            cell_area = (radius**2) * sin_term * dlon_rad
            area_2d[i, j] = np.abs(cell_area)

    return area_2d

    
def area_weight_r_sinlat(lat_values, lon_values, radius=6371000.0):
    """
    Compute approximate grid-cell areas using midpoint lat/lon.
    The lat_values, lon_values are assumed to be the midpoints of the grid boxes.
    
    We first compute the edges around these midpoints, then apply:
        area = r^2 * (sin(lat2) - sin(lat1)) * (lon2 - lon1)
    
    Parameters
    ----------
    lat_values : np.ndarray
        1D array of latitude midpoints (degrees), assumed ascending.
    lon_values : np.ndarray
        1D array of longitude midpoints (degrees), assumed ascending.
    radius : float
        Earth's radius in meters (default=6371000).
        
    Returns
    -------
    area_2d : np.ndarray
        2D array (nlat, nlon) of approximate grid-cell areas in m^2.
    """
    # 1) Convert lat/lon midpoints to radians
    lat_mid = np.deg2rad(lat_values)
    lon_mid = np.deg2rad(lon_values)

    nlat = len(lat_values)
    nlon = len(lon_values)

    # 2) Build edges from midpoints
    # We'll create nlat+1 lat-edges, nlon+1 lon-edges.
    lat_edges = np.zeros(nlat + 1)
    lon_edges = np.zeros(nlon + 1)

    # For lat edges:
    #  - The first edge is half a step below the first midpoint
    #  - The last edge is half a step above the last midpoint
    #  - For intermediate edges, we take the midpoint between adjacent lat midpoints.
    # This handles non-uniform spacing as well.
    lat_edges[0] = lat_mid[0] - 0.5*(lat_mid[1] - lat_mid[0]) if nlat > 1 else lat_mid[0] - 0.5
    for i in range(1, nlat):
        lat_edges[i] = 0.5*(lat_mid[i] + lat_mid[i - 1])
    lat_edges[-1] = lat_mid[-1] + 0.5*(lat_mid[-1] - lat_mid[-2]) if nlat > 1 else lat_mid[-1] + 0.5

    # Similarly for lon edges:
    lon_edges[0] = lon_mid[0] - 0.5*(lon_mid[1] - lon_mid[0]) if nlon > 1 else lon_mid[0] - 0.5
    for j in range(1, nlon):
        lon_edges[j] = 0.5*(lon_mid[j] + lon_mid[j - 1])
    lon_edges[-1] = lon_mid[-1] + 0.5*(lon_mid[-1] - lon_mid[-2]) if nlon > 1 else lon_mid[-1] + 0.5

    # 3) Initialize output array
    area_2d = np.zeros((nlat, nlon), dtype=float)

    # 4) Fill each cell
    for i in range(nlat):
        # sin(lat2) - sin(lat1)
        lat_south = lat_edges[i]
        lat_north = lat_edges[i + 1]
        sin_term = np.sin(lat_north) - np.sin(lat_south)

        for j in range(nlon):
            # (lon2 - lon1)
            lon_west = lon_edges[j]
            lon_east = lon_edges[j + 1]
            lon_term = lon_east - lon_west

            cell_area = (radius**2) * sin_term * lon_term
            area_2d[i, j] = np.abs(cell_area)

    return area_2d

def neutral(lat_values, lon_values):
    return np.ones((len(lat_values), len(lon_values)))

def get_weights( w_function, lats, lons ):
    weights = w_function( lats, lons )
    return xr.DataArray(
        data=weights,
        dims=('lat', 'lon'),
        coords={'lat': lats, 'lon': lons},
        name='weight'
    )