import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time
from SpeedPlannerHelpers import calculate_max_speed, interpolate_smoth_speeds

sys.path.append(os.path.abspath(".."))
from utils.map_manager import div_tray_distancia


def calculo_perfil_velocidad(trayectoria: np.ndarray, max_speed: float = 20.0, dist_sect: float = 10, G: float = 9.8,
                             C: float = 0.5, PLOT: bool = False, DEBUG: bool = False):
    """
    .. description::
    Genera un perfil de velocidad basado en la curvatura de cada tramo. Los tramos se dividiran en trozos de la misma
    distancia, generando asi perfiles de velocidad mas seguros. Se usara la siguiente formula para el calculo de la
    velocidad maxima:
        V = sqrt(R * G * C)

    .. inputs::
    :param trayectoria:                 Vector de coordenadas [x,y] que se usara para generar el perfil de velocidad.
                                        El último y el primero deben de ser diferentes.
    :type trayectoria:                  np.ndarray
    :param max_speed:                   Velocidad maxima que se le asignara a un punto (m/s)
    :type max_speed:                    float
    :param dist_sect:                   Distancia de cada sector en metros (m)
    :type dist_sect:                    float
    :param G:                           Aceleración lateral maxima del vehículo (m/s^2)
    :type G:                            float
    :param C:                           Constante de reducción para el calculo de la velocidad
    :type C:                            float
    :param PLOT:                        flag que indica si se quiere mostrar el resultado de forma grafica
    :type PLOT:                         bool
    :param DEBUG:                       flag que indica si queremos que se muestre por consola información del proceso.
    :type DEBUG:                        bool

    .. outputs::
    :return x_new_linear:               Posiciones x de las velocidades. Relacionadas con el indice de cada punto
    rtype x_new_linear:                 np.ndarray
    :return velocities_smooth_linear:   Velocidades optimizadas y interpoladas, la lista de x_new_linear son sus indices
    :rtype velocities_smooth_linear:    np.ndarray

    .. notes::
    El último de los puntos de la trayectoria y el primero deben de ser diferentes, es decir, la pista no debe estar
    cerrada.
    """
    tiempo_inicial = time.time()
    num_points = trayectoria.shape[0]

    # Dividir trayectoria en sectores del mismo tamaño
    num_sectors, sector_index = div_tray_distancia(trayectoria, dist_sect)

    # Calcula la velocidad máxima permitida para cada sector
    max_speeds = np.array([])
    for i in range(num_sectors - 1):
        max_speeds = np.hstack((max_speeds, calculate_max_speed(trayectoria[sector_index[i]:sector_index[i + 1], 0],
                                                                trayectoria[sector_index[i]:sector_index[i + 1], 1],
                                                                max_speed, G, C)))
    max_speeds = np.hstack(
        (max_speeds, calculate_max_speed(trayectoria[sector_index[i]:, 0], trayectoria[sector_index[i]:, 1],
                                         max_speed, G, C)))
    # Interpola la velocidad máxima permitida para cada punto de la pista
    velocities_smooth_linear, x_new_linear, x_old_linear = interpolate_smoth_speeds(max_speeds, sector_index,
                                                                                    num_points,
                                                                                    int_type='linear')
    if PLOT:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 8), sharex=False)
        for i in range(num_sectors):
            if i >= num_sectors - 1:  # Ultimo sector, hay que unirlo con el final
                x = np.array([trayectoria[sector_index[i], 0], trayectoria[0, 0]])
                y = np.array([trayectoria[sector_index[i], 1], trayectoria[0, 1]])
            else:  # No se el ultimo sector
                x = np.array([trayectoria[sector_index[i], 0], trayectoria[sector_index[i + 1], 0]])
                y = np.array([trayectoria[sector_index[i], 1], trayectoria[sector_index[i + 1], 1]])
            ax1.plot(x, y, 'k-')
            ax1.fill_between(x, y, y2=[y[0], y[0]], alpha=0.1)
            ax1.text(np.mean(x), np.mean(y), '{:.1f} m/s'.format(max_speeds[i]))
        ax1.plot(trayectoria[:, 0], trayectoria[:, 1], 'r--', alpha=0.5)
        ax1.set_title('Curvatura por tramos')

        ax2.set_title('Perfil de velocidad')
        ax2.set_xlabel('Tramos')

        ax2.plot(x_new_linear, velocities_smooth_linear, '-r', label='velocidades suavizadas lineal')
        ax2.plot(sector_index, max_speeds, 'o', color='blue', label='Velocidad maxima de cada tramo')
        ax2.plot(num_points, max_speeds[0], 'o', color='blue')

        ax2.legend()

        plt.show()

    if DEBUG:
        tiempo_final = time.time()
        tiempo_total = tiempo_final - tiempo_inicial
        print('Tiempo de ejecución de la optimización de velocidad ', tiempo_total)

    return x_new_linear, velocities_smooth_linear


# ------------------------------------------------------------------------------------------------------------------
# PARA TESTING -----------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

def main(args=None):
    import pandas as pd

    max_speed = 20
    dist_sect = 10
    G = 9.81
    C = 0.5

    PLOT = True
    DEBUG = True

    # Carga los datos de la pista
    data = pd.read_csv('../tracks/pista_full.csv')
    trayectoria = data.values[:, 1:]
    calculo_perfil_velocidad(trayectoria, max_speed, dist_sect, G, C, PLOT, DEBUG)


if __name__ == '__main__':
    main()
