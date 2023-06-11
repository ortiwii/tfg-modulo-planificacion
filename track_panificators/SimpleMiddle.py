import numpy as np
from PathGenerator import interpolar_spline


class SimpleMiddlePath():

    def __init__(self, k=3, s=32, num_waypoints=100):
        ######################
        ###   Parametros   ###
        ######################

        # SPLINE parameters
        # K: Degree of the spline. Cubic splines are recommended.
        self.k_ = k
        # S: A smoothing condition.  The amount of smoothness is determined by satisfying the conditions
        self.s_ = s
        self.num_waypoints_ = num_waypoints
        ######################

        # Posición del eje de dirección
        self.eje_direccion = np.array([0.765, 0.00])

        # Inicializar varibles
        self.conexiones = []
        self.internalNp = None
        self.interpolacion = None
        self.conos_amarillos = None
        self.conos_azules = None
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
        Metodo que llama a la planificación de trayectoria simple del medio del carril.

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
        self.conos_naranjas_grandes_azules = conos_naranjas_grandes[0]
        self.conos_naranjas_grandes_amarillos = conos_naranjas_grandes[1]
        self.primer_punto_coche = primer_punto_coche

        # - Llamada a la planificación de la trayectoria
        waypoints, path, P, s, x, y = self.planificacion()

        ## AÑADIR EN UN FUTURO GESTION DE LA TRIANGULACIÓN AL NO TENER UN LADO

        return waypoints, path, P, s, x, y

    def preparar_lista_conos(self):
        nci = len(self.conos_azules)  # Numeros de conos interiores en la lista de conos
        col = 3
        nce = len(self.conos_amarillos)  # Numeros de conos exteriores en la lista de conos

        # Si no tenemos mas conos que k-1 no podremos interpolar una función
        if nci > self.k_ - 1 or nce > self.k_ - 1:

            if nci > nce:  # Maximo 1 mas vamos a usar
                nci = nce + 1
            elif nce > nci:  # Maximo 1 mas vamos a usar
                nce = nci + 1
            maxConosLado = max(nci, nce)
            P = np.zeros((2 * maxConosLado, col))

            for i in range(0, maxConosLado):
                if i >= nci:  # Solo quedan conos exteriores
                    P[i * 2] = self.conos_azules[nci - 1]
                    P[i * 2 + 1] = self.conos_amarillos[i]
                elif i >= nce:  # Solo quedan conos interiores
                    P[i * 2] = self.conos_azules[i]
                    P[i * 2 + 1] = self.conos_amarillos[nce - 1]
                else:  # Quedan de los 2
                    P[i * 2] = self.conos_azules[i]
                    P[i * 2 + 1] = self.conos_amarillos[i]
            return P, nci, nce
        else:
            return None, 0, 0

    def planificacion(self, ):

        self.P, nci, nce = self.preparar_lista_conos()
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

            cant_conos = max(nci, nce)
            for i in range(0, cant_conos):
                interior = self.P[2 * i]
                exterior = self.P[2 * i + 1]
                internal.append(np.array([(interior[0] + exterior[0]) / 2, (interior[1] + exterior[1]) / 2]))
                self.conexiones.append([2 * i, 2 * i + 1])

            self.internalNp = np.array(internal)

            self.interpolacion = interpolar_spline(self.k_, self.s_, self.num_waypoints_, self.internalNp)

            if self.interpolacion is not None:
                return self.internalNp, self.interpolacion, self.P, self.conexiones, self.xBounds, self.yBounds
            else:
                print('No se ha podido hacer la planificación correctamente, se necesitan un minimo de ' + str(
                    self.k_ + 1) + ' waypoints y se tienen solo ' + str(self.internalNp.shape[0]))
                return None, None, None, None, None, None
        else:
            return None, None, np.array([]), None, None, None
def main(args=None):

    import sys
    import os
    sys.path.append(os.path.abspath(".."))
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from utils.map_manager import load_pickle_map, calcular_tramos_por_distancia


    # ------------------------------------------------------------------------------------------------------------------
    # PARA TESTING -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # - Pistas ---------
    file_path = '../tracks/pista-ordenada-00.map'  # Pista ordenada sin trayectoria definida

    # - Cargar mapa ----
    mapa = load_pickle_map(file_path)

    # - Cargar planificador
    planificador = SimpleMiddlePath(3, 32, 100)

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

    # Iteramos todos los tramos
    for tramo_act in range(len(tramos_azul) + 1):
        if tramo_act == len(tramos_azul):  # Ultimo tramo
            conos_azules = mapa['azules'][tramos_azul[-1][1]:]
            conos_azules = np.append(conos_azules, [mapa['azules'][0]], axis=0)
            conos_amarillos = mapa['amarillos'][tramos_amarillo[-1][1]-1:]
            conos_amarillos = np.append(conos_amarillos, [mapa['amarillos'][0]], axis=0)
        else:  # Resto de tramos
            conos_azules = mapa['azules'][tramos_azul[tramo_act][0]:tramos_azul[tramo_act][1] + 1]
            conos_amarillos = mapa['amarillos'][tramos_amarillo[tramo_act][0]:tramos_amarillo[tramo_act][1] + 1]

        waypoints, path, P, s, x, y = planificador.planficar_trayectoria(conos_azules, conos_amarillos,
                                                                         mapa['naranjas'],
                                                                         mapa['naranjas_grandes'], False)
        if waypoints is not None:
            # Crear la figura y los subplots con tamaño relativo

            ax1.plot(path[0], path[1], '-', color='black', label='Path')
            # ax1.triplot(P[:, 0], P[:, 1], s, color='#9c259a')
            ax1.plot(waypoints[:, 0], waypoints[:, 1], '.', color='red', markersize=6, label='Waypoints')
            waypoints_global.set_xdata(waypoints[:, 0])
            waypoints_global.set_ydata(waypoints[:, 1])

            ax2.clear()
            ax2.set_title('Actual')

            ax2.plot(path[0], path[1], '-', color='black', label='Path')
            # ax2.triplot(P[:, 0], P[:, 1], s, color='#9c259a')

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
    plt.pause(5.0)
if __name__ == '__main__':
    main()