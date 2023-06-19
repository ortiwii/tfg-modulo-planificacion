import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d # Para hacer la interpolación del perfil de velocidad

def curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denominator = (dx**2 + dy**2)**1.5
    denominator[denominator < 1e-10] = 1e-10  # evitar división por cero
    curvature = (dx*ddy - dy*ddx) / denominator
    return curvature
def get_max_speed(R, G=9.8, C=0.5):
    return np.sqrt(R * G * C)
    # V = sqrt(μ * g * R)

def calculate_max_speed(x, y, max_speed):
    c = curvature(x, y)
    avg_curvature = abs(np.mean(c))
    v_speed = get_max_speed(1 / avg_curvature)
    if v_speed > max_speed:
        return max_speed
    return v_speed

def interpolate_smoth_speeds (max_velocities, x, num_points, int_type='linear'):

    x = x+[num_points]
    max_velocities = max_velocities+[max_velocities[0]]
    f = interp1d(x, max_velocities, kind=int_type)


    x_new = np.arange(0, num_points, 1)
    velocities_smooth = f(x_new)
    return velocities_smooth, x_new, x

def main(args=None):
    import sys
    import os
    sys.path.append(os.path.abspath(".."))
    from utils.PathGenerator import interpolar_spline
    import pandas as pd

    max_speed = 20

    # Paso 1: Carga los datos de la pista
    data = pd.read_csv('../tracks/pista_full.csv')
    num_points = len(data)
    # Paso 2: Calcula la longitud de cada sector
    sector_length = 10  # metros
    total_distance = np.sum(np.sqrt(np.diff(data['x']) ** 2 + np.diff(data['y']) ** 2))
    num_sectors = int(np.ceil(total_distance / sector_length))
    sector_index = []

    # Paso 3: Divide la pista en sectores
    start_points = []
    end_points = []
    distance = 0
    for i in range(num_sectors):
        first_index = np.argmin(
            np.abs(distance - np.cumsum(np.sqrt(np.diff(data['x']) ** 2 + np.diff(data['y']) ** 2))))
        sector_index.append(first_index)
        start_point = data.iloc[first_index, :]
        end_point = data.iloc[np.argmin(
            np.abs(distance + sector_length - np.cumsum(np.sqrt(np.diff(data['x']) ** 2 + np.diff(data['y']) ** 2)))),
                    :]
        start_points.append(start_point)
        end_points.append(end_point)
        distance += sector_length

    # Paso 4: Calcula la velocidad máxima permitida para cada sector
    max_speeds = []
    for i in range(num_sectors - 1):
        max_speeds.append(calculate_max_speed(data['x'][sector_index[i]:sector_index[i + 1]],
                                              data['y'][sector_index[i]:sector_index[i + 1]], max_speed))
    max_speeds.append(calculate_max_speed(data['x'][sector_index[i]:], data['y'][sector_index[i]:], max_speed))

    # Paso 4.5: Interpola la velocidad máxima permitida para cada punto de la pista
    # velocities_linear = interpolar_spline(1, 0, num_points, max_speeds)
    velocities_smooth_linear, x_new_linear, x_old_linear = interpolate_smoth_speeds(max_speeds, sector_index, num_points,
                                                                                    int_type='linear')
    velocities_smooth_quadratic, x_new_quadratic, x_old_quadratic = interpolate_smoth_speeds(max_speeds, sector_index, num_points,
                                                                                             int_type='quadratic')
    velocities_smooth_cubic, x_new_cubic, x_old_cubic = interpolate_smoth_speeds(max_speeds, sector_index, num_points,
                                                                                 int_type='cubic')

    points = np.transpose([sector_index+[num_points], max_speeds+[max_speeds[0]]])
    print(points)
    first_grade = interpolar_spline(1, 0, num_points, points)
    # second_grade = interpolar_spline(2, 0, num_points, points)
    # third_grade = interpolar_spline(3, 0, num_points, points)

    # Paso 5: Dibuja los perfiles de velocidad para cada sector
    # fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 8), sharex=False)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 8), sharex=False)
    for i in range(num_sectors):
        x = [start_points[i]['x'], end_points[i]['x']]
        y = [start_points[i]['y'], end_points[i]['y']]
        ax1.plot(x, y, 'k-')
        ax1.fill_between(x, y, y2=[y[0], y[0]], alpha=0.1)
        ax1.text(np.mean(x), np.mean(y), '{:.1f} m/s'.format(max_speeds[i]))
    ax1.plot(data['x'], data['y'], 'r--', alpha=0.5)
    ax1.set_title('Curvatura por tramos')
    # plt.grid(True)
    ax2.set_title('Perfil de velocidad')
    # ax2.set_xlabel('Tramos')
    # ax2.plot(np.arange(0, len(max_speeds)), max_speeds, '.')
    # ax2.plot(np.arange(0, len(max_speeds)), max_speeds, '-')
    ax2.plot(first_grade[0], first_grade[1], '-', color='red', label='linear')
    # ax2.plot(second_grade[0], second_grade[1], '-', color='blue', label='cuadratic')
    # ax2.plot(third_grade[0], third_grade[1], '-', color='orange', label='cubic')
    # ax2.plot(x_new_cubic, velocities_smooth_cubic, '-', color='orange', label='velocidades suavizadas cubic')
    # ax2.plot(x_new_quadratic, velocities_smooth_quadratic, '-g', label='velocidades suavizadas quadratic')
    # ax2.plot(x_new_linear, velocities_smooth_linear, '-r', label='velocidades suavizadas lineal')
    ax2.plot(x_old_linear, max_speeds + [max_speeds[-1]], 'o', label='velocidades máximas')
    # ax2.plot(x_old_linear, max_speeds + [max_speeds[-1]], '-')

    # ax2.legend()

    plt.show()


if __name__ == '__main__':
    main()