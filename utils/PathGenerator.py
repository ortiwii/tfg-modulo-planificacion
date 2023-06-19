import numpy as np
from scipy.interpolate import splprep
from scipy.interpolate import splev
import scipy.interpolate as interpolate


def interpolar_spline(k: int, s: int, num_waypoints: int, waypoints: np.ndarray) -> np.ndarray:
    # Solo se puede interpolar si tenemos mas waypoints que el nivel de k definido
    try:
        if k < waypoints.shape[0]:
            tck, u = splprep(waypoints.T, k=k, s=s)
            u = np.linspace(0, 1, num=num_waypoints, endpoint=True)
            interpolacion = splev(u, tck)
            return interpolacion
        else:
            return waypoints
    except:
        return waypoints.T

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

from utils.plot import plot_curvature


def approximate_b_spline_path(x: list,
                              y: list,
                              n_path_points: int,
                              degree: int = 3,
                              s=None,
                              ) -> tuple:
    """
    Approximate points with a B-Spline path

    Parameters
    ----------
    x : array_like
        x position list of approximated points
    y : array_like
        y position list of approximated points
    n_path_points : int
        number of path points
    degree : int, optional
        B Spline curve degree. Must be 2<= k <= 5. Default: 3.
    s : int, optional
        smoothing parameter. If this value is bigger, the path will be
        smoother, but it will be less accurate. If this value is smaller,
        the path will be more accurate, but it will be less smooth.
        When `s` is 0, it is equivalent to the interpolation. Default is None,
        in this case `s` will be `len(x)`.

    Returns
    -------
    x : array
        x positions of the result path
    y : array
        y positions of the result path
    heading : array
        heading of the result path
    curvature : array
        curvature of the result path

    """
    distances = _calc_distance_vector(x, y)

    spl_i_x = interpolate.UnivariateSpline(distances, x, k=degree, s=s)
    spl_i_y = interpolate.UnivariateSpline(distances, y, k=degree, s=s)

    sampled = np.linspace(0.0, distances[-1], n_path_points)
    return _evaluate_spline(sampled, spl_i_x, spl_i_y)


def interpolate_b_spline_path(x, y,
                              n_path_points: int,
                              degree: int = 3) -> tuple:
    """
    Interpolate x-y points with a B-Spline path

    Parameters
    ----------
    x : array_like
        x positions of interpolated points
    y : array_like
        y positions of interpolated points
    n_path_points : int
        number of path points
    degree : int, optional
        B-Spline degree. Must be 2<= k <= 5. Default: 3

    Returns
    -------
    x : array
        x positions of the result path
    y : array
        y positions of the result path
    heading : array
        heading of the result path
    curvature : array
        curvature of the result path

    """
    return approximate_b_spline_path(x, y, n_path_points, degree, s=0.0)


def _calc_distance_vector(x, y):
    dx, dy = np.diff(x), np.diff(y)
    distances = np.cumsum([np.hypot(idx, idy) for idx, idy in zip(dx, dy)])
    distances = np.concatenate(([0.0], distances))
    distances /= distances[-1]
    return distances


def _evaluate_spline(sampled, spl_i_x, spl_i_y):
    x = spl_i_x(sampled)
    y = spl_i_y(sampled)
    dx = spl_i_x.derivative(1)(sampled)
    dy = spl_i_y.derivative(1)(sampled)
    heading = np.arctan2(dy, dx)
    ddx = spl_i_x.derivative(2)(sampled)
    ddy = spl_i_y.derivative(2)(sampled)
    curvature = (ddy * dx - ddx * dy) / np.power(dx * dx + dy * dy, 2.0 / 3.0)
    return np.array(x), y, heading, curvature,


def main():
    import sys
    import os
    sys.path.append(os.path.abspath(".."))
    from utils.map_manager import load_pickle_map

    print(__file__ + " start!!")
    # way points
    # way_point_x = [-1.0, 3.0, 4.0, 2.0, 1.0]
    # way_point_y = [0.0, -3.0, 1.0, 1.0, 3.0]
    way_point_x = [0.00, 1.00, 2.00, 4.00, 6.0]
    way_point_y = [1.00, 2.00, 2.5, 1.00, 3.0]
    # file_path = '../tracks/pista-con-trayectoria-00.map'  # Pista ordenada sin trayectoria definida

    # - Cargar mapa ----
    # mapa = load_pickle_map(file_path)
    # path = mapa['waypoints'][10:18]
    # print(path)
    #
    # way_point_x = path[:, 0]
    # way_point_y = path[:, 1]

    n_course_point = 50  # sampling number

    plt.subplots()
    rax, ray, heading, curvature = approximate_b_spline_path(
        way_point_x, way_point_y, n_course_point, s=0.5)
    plt.plot(rax, ray, '-r', label="Approximated B-Spline path")
    plot_curvature(rax, ray, heading, curvature)

    plt.title("B-Spline approximation")
    plt.plot(way_point_x, way_point_y, '-og', label="way points")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")

    plt.subplots()
    rix, riy, heading, curvature = interpolate_b_spline_path(
        way_point_x, way_point_y, n_course_point)
    plt.plot(rix, riy, '-b', label="Path")
    plot_curvature(rix, riy, heading, curvature)

    plt.title("Curvatura de trayectoria")
    # plt.plot(way_point_x, way_point_y, '-og', label="way points")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    main()
