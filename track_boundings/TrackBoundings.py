import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from scipy.spatial.transform import Rotation

# ------------------------------------------------------------------------------------------------------------------
# PARAMETROS -------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

restriccionesAzul = np.array([[-2, 0.02, 0], [-2, 4, 0], [4, 4.0, 0], [4, 0.02, 0], [-2, 0.02, 0]])
restriccionesAmarillo = np.array([[-2, -0.02, 0], [-2, -4, 0], [4, -4.0, 0], [4, -0.02, 0], [-2, -0.02, 0]])
zonaSeleccionPoints = np.array([[0.0, -1.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [2.5, 4.0, 0.0],
                                [2.833333333333334, 3.726779962499649, 0],
                                [3.388888888888889, 3.1426968052735442, 0],
                                [3.944444444444445, 2.2906142364542554, 0],
                                [4.5, 0.0, 0],
                                [3.944444444444445, -2.2906142364542554, 0],
                                [3.388888888888889, -3.1426968052735442, 0],
                                [2.833333333333334, -3.726779962499649, 0],
                                [2.5, -4.0, 0.0],
                                [0.0, -1.0, 0.0]])
zonaSeleccionImaginarios = np.array([
    [2.0, -1.5, 0.0],
    [2.0, 1.5, 0.0],
    [3.5, 1.5, 0.0],
    [3.5, -1.5, 0.0],
    [2.0, -1.5, 0.0]
])
NUM_PARTICULAS = 10
MAX_DEPTH = 2


# ------------------------------------------------------------------------------------------------------------------

# - Ecuaciones --------------------
# ec1: (x+0.5)^(2)+y^(2)=25
# ec2(x)= ((3)/(2.5)) x+1
# ec3(x)= -((3)/(2.5)) x-1
# ---------------------------------

def getLadoConosNaranjasGrades(posicion: np.ndarray,
                               conos_naranjas: np.ndarray,
                               jaw: float,
                               plot: bool = False) -> tuple:
    r = Rotation.from_euler('z', -jaw, degrees=True)
    conos_azules = []
    conos_amarillos = []
    for cono_act in conos_naranjas:

        # cono_act_rot = r.apply(cono_act)
        x = cono_act[0] - posicion[0]
        y = cono_act[1] - posicion[1]
        act = [x, y, 0]
        act_rot = r.apply(act)
        if -2 <= act_rot[0] <= 4:
            if 0 < act_rot[1] <= 4:
                if act_rot[1] <= 4.0:  # Azul
                    if len(conos_azules) > 0 and conos_azules[0][0] > act_rot[0]:
                        conos_azules = [act_rot, conos_azules[0]]
                    else:
                        conos_azules.append(cono_act)
            elif act_rot[0] >= -4:
                if act_rot[1] >= -4.0:  # Amarillo
                    if len(conos_amarillos) > 0 and conos_amarillos[0][0] > act_rot[0]:
                        conos_amarillos = [act_rot, conos_amarillos[0]]
                    else:
                        conos_amarillos.append(cono_act)

    conos_azules = np.array(conos_azules)
    conos_amarillos = np.array(conos_amarillos)
    if plot:
        zonaSeleccionAzul = r.apply(restriccionesAzul)
        zonaSeleccionAzul = zonaSeleccionAzul + posicion

        zonaSeleccionAmarillo = r.apply(restriccionesAmarillo)
        zonaSeleccionAmarillo = zonaSeleccionAmarillo + posicion

        plt.plot(conos_azules[:, 0], conos_azules[:, 1], '.', markersize=20, markerfacecolor='orange',
                 markeredgecolor='black')
        plt.plot(conos_amarillos[:, 0], conos_amarillos[:, 1], '.', markersize=20, markerfacecolor='orange',
                 markeredgecolor='black')
        plt.plot(zonaSeleccionAzul[:, 0], zonaSeleccionAzul[:, 1], '-', color='blue')
        plt.plot(zonaSeleccionAmarillo[:, 0], zonaSeleccionAmarillo[:, 1], '-', color='yellow')

    return conos_azules, conos_amarillos


def rotar_zona_selección_conos(jaw, reference, plot=False, color='blue', zona_seleccion=zonaSeleccionPoints):
    r = Rotation.from_euler('z', jaw, degrees=True)
    zonaSeleccionPointsRot = r.apply(zona_seleccion)
    zonaSeleccionPointsRot = zonaSeleccionPointsRot + reference
    if plot:
        plt.plot(zonaSeleccionPointsRot[:, 0], zonaSeleccionPointsRot[:, 1], '-', color='#FFC0CB')
        plt.plot(reference[0], reference[1], '.', markerfacecolor=color, markeredgecolor='black', markersize=15)
    return zonaSeleccionPointsRot


def isNextCone(actCone: np.ndarray,
               posibleNext: np.ndarray,
               jaw: float) -> bool:
    r = Rotation.from_euler('z', -jaw, degrees=True)
    p = [posibleNext[0] - actCone[0], posibleNext[1] - actCone[1], 0]
    pNew = r.apply(p)

    x = pNew[0]
    y = pNew[1]

    if 0 <= x < 2.5:
        return ((3 * x) / 2.5 + 1) >= y >= ((-3 * x) / 2.5 - 1)
    elif 2.5 <= x <= 4.5:
        return (x + 0.5) ** 2 + y ** 2 <= 25
    else:
        return False


def get_jaw(coneAct, reference):
    if (coneAct[0] - reference[0]) < 0:
        jaw = 180 + math.degrees(math.atan((reference[1] - coneAct[1]) / (reference[0] - coneAct[0])))
    elif (coneAct[0] - reference[0]) != 0:
        jaw = math.degrees(math.atan((reference[1] - coneAct[1]) / (reference[0] - coneAct[0])))
    else:
        if (coneAct[1] - reference[1]) < 0:
            jaw = -90
        elif (coneAct[1] - reference[1]) > 0:
            jaw = 90
        else:
            jaw = 0
    return jaw


# Las listas tienen que ser de NUMPY,  # [x,y,z]
def contruccion_recursiva(trackedCones: np.ndarray,
                          untrackedCones: np.ndarray,
                          reference: np.ndarray,
                          jaw: float,
                          plot: bool = False,
                          color: str = 'blue') -> tuple:
    i = 0
    # - Caso final ------------------
    if untrackedCones.shape[0] == 0:
        if plot:
            plt.plot(reference[0], reference[1], '.', markersize=15, markerfacecolor=color, markeredgecolor='black')
        return trackedCones, untrackedCones
    # -------------------------------

    # - Quedan mas conos ------------
    flag_encontrado = False

    # - Recorred todos los conos ----
    while i < untrackedCones.shape[0] and not flag_encontrado:
        coneAct = untrackedCones[i]
        if isNextCone(reference, coneAct, jaw):  # Encontrado el siguiente cono
            flag_encontrado = True
            untrackedCones = np.delete(untrackedCones, i, axis=0)
            trackedCones = np.append(trackedCones, np.array([coneAct]), axis=0)
            jaw_act = get_jaw(coneAct, reference)
            rotar_zona_selección_conos(jaw_act, reference, plot, color)
            trackedCones, untrackedCones = contruccion_recursiva(trackedCones, untrackedCones,
                                                                 coneAct, jaw_act,
                                                                 plot, color)

        else:  # No es el siguiente cono
            i += 1
    if not flag_encontrado and untrackedCones.shape[
        0] > 1:  # No se ha encontrado el siguiente, probar con generación de particulas aleatorias
        trackedCones_imag, untrackedCones_imag, particles, flag_encontrado, jaw_imag = buscar_conos_imaginarios_recursivo(
            trackedCones, untrackedCones, np.array([]), 0, reference, jaw)

        if flag_encontrado:  # Se han encontrado conos imaginarios correctos :)
            rotar_zona_selección_conos(jaw, reference, plot, color)
            if plot:
                # Hay que imprimir estos conos, pero seran conos imaginarios
                plt.plot(particles[:, 0], particles[:, 1], 'o', markerfacecolor='grey', markeredgecolor='black',
                         markersize=8)
            trackedCones, untrackedCones = contruccion_recursiva(trackedCones_imag, untrackedCones_imag,
                                                                 trackedCones_imag[-1], jaw_imag,
                                                                 plot, color)
    return trackedCones, untrackedCones


def buscar_conos_imaginarios_recursivo(trackedCones: np.ndarray,
                                       untrackedCones: np.ndarray,
                                       particles: np.ndarray,
                                       depth: int,
                                       reference: np.ndarray,
                                       jaw: float) -> tuple:
    # - Caso final, conos no encontrados ------------------
    if untrackedCones.shape[0] == 0 or depth >= MAX_DEPTH:
        return trackedCones, untrackedCones, particles, False, jaw
    # -------------------------------
    if (particles.shape[0] > 0):
        plt.plot(particles[:, 0], particles[:, 1], '-')
    # - Generamos los conos imaginarios dentro de la zona de seleccion ------------
    lista_conos_imaginarios = generar_puntos_aleatorios(reference, jaw)

    flag_encontrado = False

    for imag_i, cono_imaginario in enumerate(lista_conos_imaginarios):  # Recorremos todas las particulas
        # - Por cada imaginario buscamos si existe un cono siguiente dentro de los untrack
        i = 0
        jaw = get_jaw(cono_imaginario, reference)
        while i < untrackedCones.shape[0] and not flag_encontrado:
            coneAct = untrackedCones[i]
            jaw_act = get_jaw(coneAct, cono_imaginario)
            if isNextCone(cono_imaginario, coneAct, jaw):

                # - Encontrado el siguiente cono, los imaginarios previos estaban bien --------------
                flag_encontrado = True
                untrackedCones = np.delete(untrackedCones, i, axis=0)
                if particles.shape[0] > 0:
                    particles = np.append(particles, np.array([cono_imaginario]), axis=0)
                else:
                    particles = np.array([cono_imaginario])
                trackedCones = np.append(trackedCones, particles, axis=0)
                trackedCones = np.append(trackedCones, np.array([coneAct]), axis=0)
                return trackedCones, untrackedCones, particles, flag_encontrado, jaw_act
            else:  # - No es el siguiente cono -------------------------------------------------------
                i += 1
        # Generamos mas conos imaginarios en esa zona de selección, si entra aqui es que no se han encontrado
        if depth > 0:
            particles_act = np.append(particles, np.array([cono_imaginario]), axis=0)
        else:
            particles_act = np.array([cono_imaginario])
        jaw_act = get_jaw(cono_imaginario, reference)
        trackedCones_imag, untrackedCones_imag, particles_imag, flag_encontrado_imag, jaw_imag = buscar_conos_imaginarios_recursivo(
            trackedCones, untrackedCones, particles_act, depth + 1, cono_imaginario, jaw_act)
        if flag_encontrado_imag:
            return trackedCones_imag, untrackedCones_imag, particles_imag, flag_encontrado_imag, jaw_imag
    return trackedCones, untrackedCones, particles, False, jaw


def generar_puntos_aleatorios(referencia: np.ndarray, jaw: float, num_puntos: int = NUM_PARTICULAS):
    zonaSeleccionPointsRotado = rotar_zona_selección_conos(jaw, referencia, zona_seleccion=zonaSeleccionImaginarios)
    x_min = np.min(zonaSeleccionPointsRotado[:, 0])
    x_max = np.max(zonaSeleccionPointsRotado[:, 0])
    y_min = np.min(zonaSeleccionPointsRotado[:, 1])
    y_max = np.max(zonaSeleccionPointsRotado[:, 1])

    aleatorios = []
    while len(aleatorios) < num_puntos:
        aleatorios_x = np.random.uniform(x_min, x_max)
        aleatorios_y = np.random.uniform(y_min, y_max)
        punto = [aleatorios_x, aleatorios_y, 0.0]

        path = mpath.Path(zonaSeleccionPointsRotado[:, :2])
        if path.contains_point(punto[:2]):
            aleatorios.append(punto)
            # plt.plot(punto[0], punto[1], '.')
            # plt.plot([referencia[0], punto[0]], [referencia[1], punto[1]], ':', color='black')

    puntos = np.array(aleatorios)
    return puntos


def construir_pista_completa(blue_cones: np.ndarray,
                             yellow_cones: np.ndarray,
                             big_orange_cones: np.ndarray,
                             orange_cones: np.ndarray,
                             posicion: np.ndarray = np.zeros(3),
                             jaw: float = 0.00,
                             plot=True) -> tuple:

    # plt.xlim([15,40])
    # plt.ylim([-5,15])
    print('\n# 0 -- Eligiendo lado de conos naranjas grandes ----------------')

    conos_naranjas_azules, conos_naranjas_amarillos = getLadoConosNaranjasGrades(posicion,
                                                                                 big_orange_cones,
                                                                                 jaw, plot)
    print('    Elegidos los conos naranjas')
    print('\n# 1 -- Contruyendo lado amarillo -------------------------------')
    track_yellow, untrack_yellow = contruccion_recursiva(conos_naranjas_amarillos,
                                                         yellow_cones,
                                                         conos_naranjas_amarillos[0], 0,
                                                         plot, 'yellow')
    print('    Construido el lado amarillo ------------------------------')
    print('\n# 2 -- Construyendo lado azul')
    track_blue, untrack_blue = contruccion_recursiva(conos_naranjas_azules,
                                                     blue_cones,
                                                     conos_naranjas_azules[0], 0,
                                                     plot, 'blue')
    print('    Construido el lado azul')

    if plot:
        # plt.plot(untrack_blue[:, 0], untrack_blue[:, 1], 'x', markersize=10, color='red', label='Desconocidos')
        plt.plot(untrack_blue[:, 0], untrack_blue[:, 1], 'x', markersize=10, color='red')
        plt.plot(untrack_yellow[:, 0], untrack_yellow[:, 1], 'x', markersize=15, color='red')
        plt.plot([], [], 'o', markersize=8, markerfacecolor='orange', markeredgecolor='black', label='Naranjas')
        plt.plot([], [], 'o', markersize=8, markerfacecolor='blue', markeredgecolor='black', label='Azules')
        plt.plot([], [], 'o', markersize=8, markerfacecolor='yellow', markeredgecolor='black', label='Amarillos')
        plt.plot([], [], 'o', markerfacecolor='grey', markeredgecolor='black', markersize=8, label='Imaginarios')
        plt.legend()
        # plt.title('Contrucción Bordes de Pista')
        plt.show()

    return track_yellow, untrack_yellow, track_blue, untrack_blue


if __name__ == '__main__':
    from utils.map_manager import load_pickle_map, save_pickle_map
    import sys
    import os
    import pickle

    sys.path.append(os.path.dirname(__file__))

    # ------------------------------------------------------------------------------------------------------------------
    # PARA TESTING -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # - Pistas ---------
    file_path = '../tracks/pista-00.map'  # Pista desordenada sin trayectoria definida
    # file_path = '../tracks/pista-1-vacios-00.map'  # Pista no ordenada con conos que faltan y sin trayectoria definida
    # file_path = '../tracks/pista-2-vacios-00.map'  # Pista no ordenada con conos que faltan y sin trayectoria definida

    # - Cargar mapa ----
    mapa = load_pickle_map(file_path)

    # - PLOT -----------
    PLOT_TRACK = True
    PLOT_CONSTRUCCION = True

    if PLOT_TRACK:
        figure_0, axes_0 = plt.subplots()
        axes_0.plot(mapa['azules'][:, 0], mapa['azules'][:, 1], 'o', markerfacecolor='blue', markeredgecolor='black',
                    label='Conos azules')
        axes_0.plot(mapa['amarillos'][:, 0], mapa['amarillos'][:, 1], 'o', markerfacecolor='yellow',
                    markeredgecolor='black', label='Conos amarillos')
        axes_0.plot(mapa['naranjas_grandes'][:, 0], mapa['naranjas_grandes'][:, 1], 'o', markerfacecolor='orange',
                    markeredgecolor='black', label='Conos naranjas grandes')
        axes_0.arrow(-0.5, 0, 1.5, 0, head_width=0.9, head_length=0.9, ec='black')

        for i, coord in enumerate(mapa['azules']):
            axes_0.annotate(str(i), (coord[0], coord[1]), textcoords="offset points", xytext=(5, 5), ha='center')
        for i, coord in enumerate(mapa['amarillos']):
            axes_0.annotate(str(i), (coord[0], coord[1]), textcoords="offset points", xytext=(5, 5), ha='center')
        for i, coord in enumerate(mapa['naranjas_grandes']):
            axes_0.annotate(str(i), (coord[0], coord[1]), textcoords="offset points", xytext=(5, 5), ha='center')

        axes_0.set_title('Skidpad layout')
        axes_0.axis('equal')
        axes_0.legend()
        figure_0.show()

    # ------------------


    # - Construcción de la pista
    track_yellow, untrack_yellow, track_blue, untrack_blue = construir_pista_completa(mapa['azules'], mapa['amarillos'],
                                                                                      mapa['naranjas_grandes'],
                                                                                      np.array([[]]),
                                                                                      plot=PLOT_CONSTRUCCION)
    mapa['azules'] = track_blue
    mapa['amarillos'] = track_yellow
    mapa['naranjas'] = np.array([])
    file_path = '../tracks/pista-ordenada-00.map'
    save_pickle_map(mapa, file_path)

    # num = 15
    # track_blue = track_blue[2:]
    # track_yellow = track_yellow[2:]
    # track_blue = np.delete(track_blue, [8,9, 22,23], axis=0)
    # track_yellow = np.delete(track_yellow, [12, 13, 25], axis=0)
    # # print(track_blue)
    # #
    # track_yellow, untrack_yellow, track_blue, untrack_blue = construir_pista_completa(track_blue, track_yellow,
    #                                                                                   mapa['naranjas_grandes'],
    #                                                                                   np.array([[]]),
    #                                                                                   plot=True)