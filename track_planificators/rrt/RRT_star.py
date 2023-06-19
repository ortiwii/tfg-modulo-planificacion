import math
import random
import matplotlib.pyplot as plt
import numpy as np
from .RRT import RRT
import sys
import os
sys.path.append(os.path.abspath("../.."))
from utils.PathGenerator import interpolar_spline, approximate_b_spline_path

class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area_x,
                 rand_area_y,
                 expand_dis=30.0,
                 path_resolution=1.0,
                 goal_sample_rate=20,
                 max_iter=300,
                 connect_circle_dist=50.0,
                 search_until_max_iter=False,
                 robot_radius=0.0):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        super().__init__(start, goal, obstacle_list, rand_area_x, rand_area_y, expand_dis,
                         path_resolution, goal_sample_rate, max_iter,
                         robot_radius=robot_radius)
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal[0], goal[1])
        self.search_until_max_iter = search_until_max_iter
        self.node_list = []

    def planning(self, animation=True):
        """
        rrt star path planning

        animation: flag for animation on or off .
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            # print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd,
                                  self.expand_dis)
            near_node = self.node_list[nearest_ind]
            new_node.cost = near_node.cost + \
                            math.hypot(new_node.x - near_node.x,
                                       new_node.y - near_node.y)

            if self.check_collision(
                    new_node, self.obstacle_list, self.robot_radius):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(
                    new_node, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)

            if animation:
                self.draw_graph(rnd)

            if ((not self.search_until_max_iter)
                    and new_node):  # if reaches goal
                last_index = self.search_best_goal_node()
                if last_index is not None:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index)

        return None

    def choose_parent(self, new_node, near_inds):
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
            Arguments:
            --------
                new_node, Node
                    randomly generated node with a path from its neared point
                    There are not coalitions between this node and th tree.
                near_inds: list
                    Indices of indices of the nodes what are near to new_node

            Returns.
            ------
                Node, a copy of new_node
        """
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(
                    t_node, self.obstacle_list, self.robot_radius):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [
            self.calc_dist_to_goal(n.x, n.y) for n in self.node_list
        ]
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list
            if i <= self.expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_collision(
                    t_node, self.obstacle_list, self.robot_radius):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        safe_goal_costs = [self.node_list[i].cost +
                           self.calc_dist_to_goal(self.node_list[i].x, self.node_list[i].y)
                           for i in safe_goal_inds]

        min_cost = min(safe_goal_costs)
        for i, cost in zip(safe_goal_inds, safe_goal_costs):
            if cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        """
        1) defines a ball centered on new_node
        2) Returns all nodes of the three that are inside this ball
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes inside the ball of
                    radius r
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt(math.log(nnode) / nnode)
        # if expand_dist exists, search vertices in a range no more than
        # expand_dist
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2
                     for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
        return near_inds

    def rewire(self, new_node, near_inds):
        """
            For each node in near_inds, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.
            Parameters:
            ----------
                new_node, Node
                    Node randomly added which can be joined to the tree

                near_inds, list of uints
                    A list of indices of the self.new_node which contains
                    nodes within a circle of a given radius.
            Remark: parent is designated in choose_parent.

        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(
                edge_node, self.obstacle_list, self.robot_radius)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                for node in self.node_list:
                    if node.parent == self.node_list[i]:
                        node.parent = edge_node
                self.node_list[i] = edge_node
                self.propagate_cost_to_leaves(self.node_list[i])

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)


class RRTStarPlanificator():
    def __init__(self,
                 expand_dis,
                 robot_radius,
                 max_iter,
                 path_resolution,
                 search_until_max_iter=False):
        # - Para la interpolación
        self.k_ = 3
        self.s_ = 4
        self.num_points = 50

        self.expand_dis = expand_dis
        self.robot_radius = robot_radius
        self.max_iter = max_iter
        self.path_resolution = path_resolution
        self.search_until_max_iter = search_until_max_iter

        self.radio_conos = 1.0
        self.conos_azules = None
        self.conos_amarillos = None
        self.conos_naranjas = None
        self.conos_naranjas_grandes_azules = None
        self.conos_naranjas_grandes_amarillos = None

    def planficar_trayectoria(self, conos_azules: np.ndarray,
                              conos_amarillos: np.ndarray,
                              conos_naranjas: np.ndarray,
                              conos_naranjas_grandes: np.ndarray,
                              ) -> tuple:
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

        Retorna:
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

        lado_azul = np.array(interpolar_spline(self.k_, self.s_, self.num_points, self.conos_azules[:, :2])).T
        lado_amarillo = np.array(interpolar_spline(self.k_, self.s_, self.num_points, self.conos_amarillos[:, :2])).T

        # lado_azul = np.array(approximate_b_spline_path(self.conos_azules[:, 0], self.conos_azules[:,1], self.num_points, self.k_, s=self.s_))
        xMaxAzul, yMaxAzul = conos_azules[:, :2].max(axis=0)
        xMinAzul, yMinAzul = conos_azules[:, :2].min(axis=0)

        xMaxAmarillo, yMaxAmarillo = conos_amarillos[:, :2].max(axis=0)
        xMinAmarillo, yMinAmarillo = conos_amarillos[:, :2].min(axis=0)

        maxX, minX = max(xMaxAzul, xMaxAmarillo), min(xMinAzul, xMinAmarillo)
        maxY, minY = max(yMaxAzul, yMaxAmarillo), min(yMinAzul, yMinAmarillo)

        lista_conos = np.concatenate([lado_azul[:, :2], lado_amarillo[:, :2]], axis=0)
        lista_conos_radio = np.column_stack((lista_conos, np.full(lista_conos.shape[0], self.radio_conos)))

        start_point = np.array([
            (conos_azules[0][0] + conos_amarillos[0][0]) / 2,  # x
            (conos_azules[0][1] + conos_amarillos[0][1]) / 2  # y
        ])

        goal_point = np.array([
            (conos_azules[-1][0] + conos_amarillos[-1][0]) / 2,  # x
            (conos_azules[-1][1] + conos_amarillos[-1][1]) / 2  # y
        ])
        # Set Initial parameters
        rrt_star = RRTStar(
            start=start_point,
            goal=goal_point,
            rand_area_x=[minX, maxX],
            rand_area_y=[minY, maxY],
            obstacle_list=lista_conos_radio,
            expand_dis=self.expand_dis,
            robot_radius=self.robot_radius,
            max_iter=self.max_iter,
            path_resolution=self.path_resolution,
            search_until_max_iter=self.search_until_max_iter)

        waypoints = np.array(rrt_star.planning(animation=False))
        path = np.array(interpolar_spline(3, 4, 50, np.array(waypoints))).T
        # path = np.transpose(approximate_b_spline_path(waypoints[:,0], waypoints[:,1], 50, 4))
        return waypoints, path


def main():
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
    ax1.plot([], [], '.', color='red', markersize=6, label='Waypoints')

    ax2.set_title('Actual')

    # Lo vamos a separar por tramos para hacer una simulación de planificación local
    num_tramos = 10
    max_iter = 300

    tramos_azul = calcular_tramos_por_distancia(mapa['azules'], num_tramos)
    tramos_amarillo = calcular_tramos_por_distancia(mapa['amarillos'], num_tramos)
    for tramo_act in range(len(tramos_azul) + 1):
        if tramo_act == len(tramos_azul):  # Ultimo tramo
            conos_azules = mapa['azules'][tramos_azul[-1][1]:]
            conos_azules = np.append(conos_azules, [mapa['azules'][0]], axis=0)
            conos_amarillos = mapa['amarillos'][tramos_amarillo[-1][1]:]
            conos_amarillos = np.append(conos_amarillos, [mapa['amarillos'][0]], axis=0)

        else:  # Resto de tramos
            conos_azules = mapa['azules'][tramos_azul[tramo_act][0]:tramos_azul[tramo_act][1] + 1]
            conos_amarillos = mapa['amarillos'][tramos_amarillo[tramo_act][0]:tramos_amarillo[tramo_act][1] + 1]

        planificator = RRTStarPlanificator(expand_dis=0.5, robot_radius=1.8, max_iter=max_iter, path_resolution=0.05,
                                           search_until_max_iter=True)
        waypoints, path = planificator.planficar_trayectoria(conos_azules, conos_amarillos, mapa['naranjas'],
                                                             mapa['naranjas_grandes'])
        # waypoints_T = np.transpose(waypoints)
        #
        # ax2.clear()
        # ax2.set_title('Actual')
        # ax1.plot(conos_azules[:, 0], conos_azules[:, 1], '-', color='blue')
        # ax2.plot(conos_azules[:, 0], conos_azules[:, 1], '-', color='blue')
        # ax1.plot(conos_amarillos[:, 0], conos_amarillos[:, 1], '-', color='yellow')
        # ax2.plot(conos_amarillos[:, 0], conos_amarillos[:, 1], '-', color='yellow')
        # # ax1.plot(waypoints_T[0], waypoints_T[1], '-', color='black', label='Path')
        # ax1.plot(path[:, 0], path[:, 1], '-', color='red', label='Path smoth')
        # ax1.plot(waypoints_T[0], waypoints_T[1], 'o', color='black', label='Path')
        # # ax2.plot(waypoints_T[0], waypoints_T[1], '-', color='black', label='Path')
        # ax2.plot(path[:, 0], path[:, 1], '-', color='red', label='Path smoth')
        # ax2.plot(waypoints_T[0], waypoints_T[1], 'o', color='black', label='Path')
        # ax2.plot(conos_azules[:, 0], conos_azules[:, 1], 'o', markerfacecolor='blue',
        #          markeredgecolor='black', markersize=8)
        #
        # ax2.plot(conos_amarillos[:, 0], conos_amarillos[:, 1], 'o', markerfacecolor='yellow',
        #          markeredgecolor='black', markersize=8)
        #
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        #
        # plt.pause(20.0)


if __name__ == '__main__':
    main()
