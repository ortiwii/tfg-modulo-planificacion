import numpy as np
from scipy.spatial import Delaunay
import sys
import os
sys.path.append(os.path.abspath("../.."))
from utils.PathGenerator import interpolar_spline

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')))


class TriangulacionDelaunay():

    def __init__(self, k, s, num_waypoints):

        ######################
        ###   Parametros   ###
        ######################
        self.eje_direccion = np.array([0.765, 0.00])
        # SPLINE parameters
        # K: Degree of the spline. Cubic splines are recommended.
        self.k_ = k
        # S: A smoothing condition.  The amount of smoothness is determined by satisfying the conditions
        self.s_ = s
        self.num_waypoints_ = num_waypoints
        ######################

        self.internalNp = None
        self.interpolacion = None
        self.conos_azules = None
        self.conos_amarillos = None
        self.conos_naranjas = None
        self.conos_naranjas_grandes_azules = None
        self.conos_naranjas_grandes_amarillos = None
        self.primer_punto_coche = False

    def set_eje_dirección(self, x, y):
        self.eje_direccion = np.array([x, y])

    def planficar_trayectoria(self, conos_azules: np.ndarray,
                              conos_amarillos: np.ndarray,
                              conos_naranjas: np.ndarray,
                              conos_naranjas_grandes: np.ndarray,
                              primer_punto_coche: bool) -> tuple:
        """
        Metodo que llama a la planificación de trayectoria usando el planificador de Delaunay.

        Aquí, se genenera una trayectoria partiendo de una pista ya ordenada, donde todos los conos reales, imaginarios y de todos los colores
        están ordenados con su posición en la pista.

        Parámetros:
        - conos_amarillos: Lista de conos amarillos ordenados de tipo numpy de tamaño [n, 3].
        - conos_azules: Lista de conos amarillos ordenados de tipo numpy de tamaño [n, 3].
        - conos_naranjas: Lista de conos naranjas pequeños y ordenados de tipo numpy de tamaño [n, 3].
        - conos_naranjas_grandes: Lista de conos naranjas grandes ordenados de tipo numpy de tamaño [2, n, 3]. El indice 0 será para conos azules y
        el índice 1 para los conos amarillos.
        - primer_punto_coche: Bool para indicar si hay que añadir el eje de dirección

        Retorna:
        - waypoints: Lista de puntos medios de la trayectoria generada
        - path: Lista de puntos interpolada con los waypoints generados
        - P: Lista de puntos ordenada utilizada para la triangulación.
        - s: Lista de triangulaciones generadas para planificación
        - xBounds: (xMax, xMin).
        - yBounds: (yMax, yMin).
        """

        self.conos_azules = conos_azules
        self.conos_amarillos = conos_amarillos
        self.conos_naranjas = conos_naranjas
        # self.conos_naranjas_grandes_azules = conos_naranjas_grandes[0]
        # self.conos_naranjas_grandes_amarillos = conos_naranjas_grandes[1]
        self.primer_punto_coche = primer_punto_coche

        # - Llamada a la planificación de la trayectoria
        self.internalNp, self.interpolacion, self.P, self.s, self.xBounds, self.yBounds = self.triangulacion()

        ## AÑADIR EN UN FUTURO GESTION DE LA TRIANGULACIÓN AL NO TENER UN LADO

        return self.internalNp, self.interpolacion, self.P, self.s, self.xBounds, self.yBounds

    def ordenar_respecto(self, coche, lista):
        return np.array(sorted(lista, key=lambda p: (p[0] - coche[0]) ** 2 + (p[1] - coche[1]) ** 2))

    def getEdges(self, triangle, edges, isEven):
        # t0 t1
        dist_min = 10
        flag = False
        if (isEven[0] + isEven[1] == 1):
            if (abs(triangle[0] - triangle[1]) < dist_min):
                edges[triangle[0], triangle[1]] = 1
                edges[triangle[1], triangle[0]] = 1
            else:
                flag = True
        # t0 t2
        if (isEven[0] + isEven[2] == 1):
            if (abs(triangle[0] - triangle[2]) < dist_min):
                edges[triangle[0], triangle[2]] = 1
                edges[triangle[2], triangle[0]] = 1
            else:
                flag = True
        # t1 t2
        if (isEven[1] + isEven[2] == 1):
            if (abs(triangle[2] - triangle[1]) < dist_min):
                edges[triangle[1], triangle[2]] = 1
                edges[triangle[2], triangle[1]] = 1
            else:
                flag = True

        return edges, flag

    def preparar_lista_conos(self):
        nci = len(self.conos_azules)  # Numeros de conos interiores en la lista de conos
        col = 3
        nce = len(self.conos_amarillos)  # Numeros de conos exteriores en la lista de conos
        if (nci > 1 and nce > 1):

            self.maxConosLado = max(nci, nce)
            P = np.zeros((2 * self.maxConosLado, col))

            for i in range(0, self.maxConosLado):
                if (i >= nci):  # Solo quedan conos exteriores
                    P[i * 2] = self.conos_azules[nci - 1]
                    P[i * 2 + 1] = self.conos_amarillos[i]
                elif (i >= nce):  # Solo quedan conos interiores
                    P[i * 2] = self.conos_azules[i]
                    P[i * 2 + 1] = self.conos_amarillos[nce - 1]
                else:  # Quedan de los 2
                    P[i * 2] = self.conos_azules[i]
                    P[i * 2 + 1] = self.conos_amarillos[i]
            return P
        else:
            return None

    def triangulacion(self):

        self.P = self.preparar_lista_conos()
        if self.P is not None:

            internal = []
            xMaxI, yMaxI = self.conos_azules[:, :2].max(axis=0)
            xMinI, yMinI = self.conos_azules[:, :2].min(axis=0)

            xMaxE, yMaxE = self.conos_amarillos[:, :2].max(axis=0)
            xMinE, yMinE = self.conos_amarillos[:, :2].min(axis=0)

            xMax, yMax = max(xMaxI, xMaxE), max(yMaxI, yMaxE)
            xMin, yMin = min(xMinI, xMinE), min(yMinI, yMinE)

            self.xBounds = np.array([xMax, xMin])
            self.yBounds = np.array([yMax, yMin])

            if self.primer_punto_coche:  # Si es true se le añade el primer punto
                internal.append(
                    np.array(self.eje_direccion))  # Para despues hacer la trayectoria desde el morro del coche

            # edgesMatrix = np.zeros([P.shape[0], P.shape[0]])
            edgesMatrix = np.zeros([self.maxConosLado * 2, self.maxConosLado * 2])

            # Crear triangulacion
            TR = Delaunay(self.P[:, :2])

            self.s = TR.simplices
            i = 0
            # Iterar todos los triangulos creados para filtrarlos
            while i < self.s.shape[0]:

                x = self.s[i]
                isEven = x % 2
                if ((isEven[0] == 0 and isEven[1] == 0 and isEven[2] == 0) or (
                        isEven[0] == 1 and isEven[1] == 1 and isEven[2] == 1)):
                    self.s = np.delete(self.s, i, 0)
                else:
                    edgesMatrix, flag = self.getEdges(x, edgesMatrix, isEven)
                    if (flag):
                        self.s = np.delete(self.s, i, 0)
                    else:
                        i = i + 1
            for fila in range(0, self.P.shape[0]):
                for columna in range(0, fila):
                    if (edgesMatrix[fila][columna] == 1):  # Es uno interno
                        p1 = self.P[fila]
                        p2 = self.P[columna]

                        internal.append(np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]))

            self.internalNp = np.array(internal)

            self.interpolacion = interpolar_spline(self.k_, self.s_, self.num_waypoints_, self.internalNp)
            if self.interpolacion is not None:
                return self.internalNp, self.interpolacion, self.P, self.s, self.xBounds, self.yBounds
            else:
                print('No se ha podido hacer la triangulación correctamente, se necesitan un minimo de ' + str(
                    self.k_ + 1) + ' waypoints y se tienen solo ' + str(self.internalNp.shape[0]))
                return None, None, None, None, None, None
        else:
            print('No se ha podido hacer la triangulación correctamente, no hay 2 listas de conos lo suficientemente '
                  'grandes')
            return None, None, None, None, None, None


def main(args=None):
    from utils.map_manager import load_pickle_map, calcular_tramos_por_distancia
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec



    # ------------------------------------------------------------------------------------------------------------------
    # PARA TESTING -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # - Pistas ---------
    file_path = '../../tracks/pista-ordenada-00.map'  # Pista ordenada sin trayectoria definida

    # - Cargar mapa ----
    mapa = load_pickle_map(file_path)

    # - Cargar planificador
    planificador = TriangulacionDelaunay(3, 64, 100)

    # - Inicializar plot en modo interactivo
    plt.ion()
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], height_ratios=[1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ax1.set_title('Pista Completa')
    ax1.plot(mapa['azules'][:, 0], mapa['azules'][:, 1], 'o', markerfacecolor='blue',
             markeredgecolor='black', markersize=8)
    ax1.plot(mapa['amarillos'][:, 0], mapa['amarillos'][:, 1], 'o', markerfacecolor='yellow',
             markeredgecolor='black', markersize=8)
    ax1.plot(mapa['naranjas_grandes'][:, 0], mapa['naranjas_grandes'][:, 1], 'o', markerfacecolor='orange',
             markeredgecolor='black', markersize=8)
    waypoints_global, = ax1.plot([], [], '.', color='red', markersize=6, label='Waypoints')

    ax2.set_title('Actual')

    # Lo vamos a separar por tramos para hacer una simulación de planificación local
    num_tramos = 10
    tramos_azul = calcular_tramos_por_distancia(mapa['azules'], num_tramos)
    tramos_amarillo = calcular_tramos_por_distancia(mapa['amarillos'], num_tramos)
    for tramo_act in range(len(tramos_azul) + 1):
        if tramo_act == len(tramos_azul):  # Ultimo tramo
            conos_azules = mapa['azules'][tramos_azul[-1][1]:]
            conos_azules = np.append(conos_azules, [mapa['azules'][0]], axis=0)
            conos_amarillos = mapa['amarillos'][tramos_amarillo[-1][1]:]
            conos_amarillos = np.append(conos_amarillos, [mapa['amarillos'][0]], axis=0)
            waypoints, path, P, s, x, y = planificador.planficar_trayectoria(conos_azules, conos_amarillos,
                                                                             mapa['naranjas'],
                                                                             mapa['naranjas_grandes'], False)
        else: # Resto de tramos
            conos_azules = mapa['azules'][tramos_azul[tramo_act][0]:tramos_azul[tramo_act][1] + 1]
            conos_amarillos = mapa['amarillos'][tramos_amarillo[tramo_act][0]:tramos_amarillo[tramo_act][1] + 1]
            waypoints, path, P, s, x, y = planificador.planficar_trayectoria(conos_azules, conos_amarillos,
                                                                             mapa['naranjas'],
                                                                             mapa['naranjas_grandes'], False)

        if waypoints is not None:
            # Crear la figura y los subplots con tamaño relativo

            ax1.plot(path[0], path[1], '-', color='black', label='Path')
            ax1.triplot(P[:, 0], P[:, 1], s, color='#9c259a')
            ax1.plot(waypoints[:, 0], waypoints[:, 1], '.', color='red', markersize=6, label='Waypoints')
            waypoints_global.set_xdata(waypoints[:, 0])
            waypoints_global.set_ydata(waypoints[:, 1])

            ax2.clear()
            ax2.set_title('Actual')

            ax2.plot(path[0], path[1], '-', color='black', label='Path')
            ax2.triplot(P[:, 0], P[:, 1], s, color='#9c259a')

            rangoInt = range(0, P.shape[0], 2)
            ax2.plot(P[rangoInt, 0], P[rangoInt, 1], '-', color='blue')
            ax2.plot(P[rangoInt, 0], P[rangoInt, 1], 'o', markerfacecolor='blue',
                     markeredgecolor='black', markersize=8, label='Azules')

            rangoExt = range(1, P.shape[0], 2)
            ax2.plot(P[rangoExt, 0], P[rangoExt, 1], '-', color='yellow')
            ax2.plot(P[rangoExt, 0], P[rangoExt, 1], 'o', markerfacecolor='yellow',
                     markeredgecolor='black', markersize=8, label='Amarillos')

            ax2.plot(waypoints[:, 0], waypoints[:, 1], '.', color='red', markersize=8, label='Waypoints')

            fig.canvas.draw()
            fig.canvas.flush_events()

        plt.pause(0.5)


if __name__ == '__main__':
    main()
