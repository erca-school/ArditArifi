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


def area_weight_r_sinlat(lat_values, lon_values, radius=6371000.0):
    """
    Compute approximate grid-cell areas (or area weights) using:
        area = r^2 * d(lambda) * d(sin(lat))
    where r is Earth's radius.
    
    Parameters
    ----------
    lat_values : numpy.ndarray
        1D array of latitude values (degrees).
    lon_values : numpy.ndarray
        1D array of longitude values (degrees).
    radius : float
        Radius of Earth in meters (default = 6,371,000 m).
        
    Returns
    -------
    weights : 2D numpy.ndarray
        2D array (lat x lon) of computed area weights.
    """
    # Convert lat and lon to radians
    lat_rad = np.deg2rad(lat_values)
    lon_rad = np.deg2rad(lon_values)

    dlat = np.diff(lat_rad)           # difference between consecutive lats
    dlon = np.diff(lon_rad)           # difference between consecutive lons

    # We'll define area for each lat "strip" and each lon "strip"
    # area = r^2 * (sin(lat2) - sin(lat1)) * (lon2 - lon1)
    # For a 2D array, we need the areas in each "cell".
    # We'll store the result in a 2D array of shape (len(lat), len(lon)).
    area_2d = np.zeros((len(lat_values), len(lon_values)))

    # Loop to fill in the cell-based area
    for i in range(len(lat_values) - 1):
        # sin(lat2) - sin(lat1)
        sin_term = np.sin(lat_rad[i+1]) - np.sin(lat_rad[i])
        for j in range(len(lon_values) - 1):
            # (lon2 - lon1)
            lon_term = lon_rad[j+1] - lon_rad[j]
            cell_area = (radius ** 2) * sin_term * lon_term
            # We can store it for the "upper-left corner" of the cell
            area_2d[i, j] = np.abs(cell_area)

    return area_2d


def get_weights( w_function, lats, lons ):
    weights = w_function( lats, lons )
    return xr.DataArray(
        data=weights,
        dims=('lat', 'lon'),
        coords={'lat': lats, 'lon': lons},
        name='weight'
    )