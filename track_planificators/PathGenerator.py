import numpy as np
from scipy.interpolate import splprep
from scipy.interpolate import splev


def interpolar_spline(k: int, s: int, num_waypoints: int, waypoints: np.ndarray) -> np.ndarray:
    # Solo se puede interpolar si tenemos mas waypoints que el nivel de k definido
    if k < waypoints.shape[0]:
        tck, u = splprep(waypoints.T, k=k, s=s)
        u = np.linspace(0, 1, num=num_waypoints, endpoint=True)
        interpolacion = splev(u, tck)
        return interpolacion
    else:
        return None
