import numpy as np
import xarray as xr
from haversine import haversine, Unit

def area_weight_haversine(lat_values, lon_values):
    """
    Compute grid-cell area weights (normalized) using the haversine formula from Remi's script:
    https://doi.org/10.5281/zenodo.7590005
    
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