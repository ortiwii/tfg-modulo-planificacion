import numpy as np
from scipy.interpolate import interp1d  # Para hacer la interpolación del perfil de velocidad

def curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denominator = (dx ** 2 + dy ** 2) ** 1.5
    denominator[denominator < 1e-10] = 1e-10  # evitar división por cero
    curvature = (dx * ddy - dy * ddx) / denominator
    return curvature


def get_max_speed(R, G, C):
        # V = sqrt(μ * g * R)
    return np.sqrt(R * G * C)


def calculate_max_speed(x, y, max_speed, G=9.8, C=0.5):
    c = curvature(x, y)
    avg_curvature = abs(np.mean(c))
    v_speed = get_max_speed(1 / avg_curvature, G, C)
    if v_speed > max_speed:
        return max_speed
    return v_speed


def interpolate_smoth_speeds(max_velocities: np.ndarray, x: np.ndarray, num_points, int_type='linear'):
    x = np.hstack((x, num_points))
    max_velocities = np.hstack((max_velocities, max_velocities[0]))

    f = interp1d(x, max_velocities, kind=int_type)

    x_new = np.arange(0, num_points, 1)
    velocities_smooth = f(x_new)
    return velocities_smooth, x_new, x

