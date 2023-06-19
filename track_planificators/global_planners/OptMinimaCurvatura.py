import numpy as np
import math
import quadprog
# import cvxopt
import time
import matplotlib.pyplot as plt

def optimizar_minima_curvatura(reftrack: np.ndarray,
                 normvectors: np.ndarray,
                 A: np.ndarray,
                 kappa_bound: float,
                 w_veh: float,
                 print_debug: bool = False,
                 plot_debug: bool = False,
                 closed: bool = True,
                 psi_s: float = None,
                 psi_e: float = None,
                 fix_s: bool = False,
                 fix_e: bool = False) -> tuple:
    """
    .. description::
    Esta función utiliza un solucionador QP para minimizar la curvatura sumada de una trayectoria moviendo los puntos
    de la trayectoria a lo largo de sus vectores normales dentro de la anchura de la pista. La función puede utilizarse
    para trazados cerrados y no cerrados. Para trazados no cerrados, la dirección psi_s y psi_e se aplica en el primer y
    último punto del trazado. Además, en el caso de una vía no cerrada, el primer y el último punto de la referencia no
    están sujetos a optimización y permanecen iguales.

    .. inputs::
    :param reftrack:    matriz que contiene la pista de referencia, es decir, una línea de referencia y los anchos de
                        pista correspondientes a a la derecha y a la izquierda [x, y, w_tr_right, w_tr_left]
                        (la unidad es el metro, ¡debe estar sin cerrar!)
    :type reftrack:     np.ndarray
    :param normvectors: vectores normales normalizados para cada punto de la pista de referencia [x_componente,
                        y_componente]. (la unidad es el metro, ¡debe estar sin cerrar!)
    :type normvectors:  np.ndarray
    :param A:           Matriz de sistema de ecuaciones lineales para splines (aplicable tanto para la dirección x como
                        para la dirección y).
                        -> Las matrices del sistema tienen la forma a_i, b_i * t, c_i * t^2, d_i * t^3
                        -> ver calc_splines.py para más información o para obtener esta matriz
    :type A:            np.ndarray
    :param kappa_bound: límite de curvatura a considerar durante la optimización.
    :type kappa_bound:  float
    :param w_veh:       anchura del vehículo en m. Se considera durante el cálculo de las desviaciones permitidas de la
                        línea de referencia.
    :type w_veh:        float
    :param print_debug: flag bool para imprimir mensajes de depuración.
    :type print_debug:  bool
    :param plot_debug:  flag bool para trazar las curvaturas que se calculan basándose en la linealización original y
                        en una linealización alrededor de la solución.
    :type plot_debug:   bool
    :param closed:      flag bool que especifica si debe suponerse una vía cerrada o no cerrada
    :type closed:       bool
    :param psi_s:       dirección que se aplicará en el primer punto de las vías no cerradas
    :type psi_s:        float
    :param psi_e:       dirección que se aplicará en el último punto de las vías no cerradas
    :type psi_e:        float
    :param fix_s:       determina si el punto inicial está fijado a la línea de referencia para las vías no cerradas
    :type fix_s:        bool
    :param fix_e:       determina si el último punto está fijado a la línea de referencia para las vías no cerradas
    :type fix_e:        bool

    .. outputs::
    :return alpha_mincurv:  vector solución del problema optimización que contiene el desplazamiento lateral en m
                            para cada punto.
    :rtype alpha_mincurv:   np.ndarray
    :return curv_error_max: error máximo de curvatura al comparar la curvatura calculada a partir de la linealización
                            en torno a la pista de referencia original y en torno a la solución.
    :rtype curv_error_max:  float
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARACIONES ----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    no_points = reftrack.shape[0]

    no_splines = no_points
    if not closed:
        no_splines -= 1

    # comprobar entradas
    if no_points != normvectors.shape[0]:
        raise RuntimeError("El tamaño del array de reftrack debe ser el mismo que el de normvectors!")

    if (no_points * 4 != A.shape[0] and closed) or (no_splines * 4 != A.shape[0] and not closed)\
            or A.shape[0] != A.shape[1]:
        raise RuntimeError("La matriz A del sistema de ecuaciones spline tiene dimensiones incorrectas!")

    # crear matriz de extracción -> sólo se necesitan los coeficientes b_i del sistema de ecuaciones lineales resuelto
    # para el gradiente

    # información
    A_ex_b = np.zeros((no_points, no_splines * 4), dtype=int)

    for i in range(no_splines):
        A_ex_b[i, i * 4 + 1] = 1    # 1 * b_ix = E_x * x

    # coeficientes para el final del spline (t = 1)
    if not closed:
        A_ex_b[-1, -4:] = np.array([0, 1, 2, 3])

    # crear matriz de extracción -> sólo se necesitan los coeficientes c_i del sistema de ecuaciones lineales resuelto
    # para la curvatura

    # información
    A_ex_c = np.zeros((no_points, no_splines * 4), dtype=int)

    for i in range(no_splines):
        A_ex_c[i, i * 4 + 2] = 2    # 2 * c_ix = D_x * x

    # coeficientes para el final del spline (t = 1)
    if not closed:
        A_ex_c[-1, -4:] = np.array([0, 0, 2, 6])

    # invertir la matriz A resultante del sistema de ecuaciones lineales de la configuración spline y aplicar la matriz
    # de extracción
    A_inv = np.linalg.inv(A)
    T_c = np.matmul(A_ex_c, A_inv)

    # configurar las matrices M_x y M_y incluyendo la información del gradiente, es decir, poner los vectores normales
    # en forma de matriz
    M_x = np.zeros((no_splines * 4, no_points))
    M_y = np.zeros((no_splines * 4, no_points))

    for i in range(no_splines):
        j = i * 4

        if i < no_points - 1:
            M_x[j, i] = normvectors[i, 0]
            M_x[j + 1, i + 1] = normvectors[i + 1, 0]

            M_y[j, i] = normvectors[i, 1]
            M_y[j + 1, i + 1] = normvectors[i + 1, 1]
        else:
            M_x[j, i] = normvectors[i, 0]
            M_x[j + 1, 0] = normvectors[0, 0]  # cerrar spline

            M_y[j, i] = normvectors[i, 1]
            M_y[j + 1, 0] = normvectors[0, 1]

    # configurar las matrices q_x y q_y incluyendo la información de las coordenadas del punto
    q_x = np.zeros((no_splines * 4, 1))
    q_y = np.zeros((no_splines * 4, 1))

    for i in range(no_splines):
        j = i * 4

        if i < no_points - 1:
            q_x[j, 0] = reftrack[i, 0]
            q_x[j + 1, 0] = reftrack[i + 1, 0]

            q_y[j, 0] = reftrack[i, 1]
            q_y[j + 1, 0] = reftrack[i + 1, 1]
        else:
            q_x[j, 0] = reftrack[i, 0]
            q_x[j + 1, 0] = reftrack[0, 0]

            q_y[j, 0] = reftrack[i, 1]
            q_y[j + 1, 0] = reftrack[0, 1]

    # para las vías no cerradas, especifique las restricciones de inicio y fin de la vía
    if not closed:
        q_x[-2, 0] = math.cos(psi_s + math.pi / 2)
        q_y[-2, 0] = math.sin(psi_s + math.pi / 2)

        q_x[-1, 0] = math.cos(psi_e + math.pi / 2)
        q_y[-1, 0] = math.sin(psi_e + math.pi / 2)

    # establecer las matrices P_xx, P_xy, P_yy
    x_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_x)
    y_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_y)

    x_prime_sq = np.power(x_prime, 2)
    y_prime_sq = np.power(y_prime, 2)
    x_prime_y_prime = -2 * np.matmul(x_prime, y_prime)

    curv_den = np.power(x_prime_sq + y_prime_sq, 1.5)                   # calcular el denominador de curvatura
    curv_part = np.divide(1, curv_den, out=np.zeros_like(curv_den),
                          where=curv_den != 0)                          # dividir donde no sea cero (elementos diag)
    curv_part_sq = np.power(curv_part, 2)

    P_xx = np.matmul(curv_part_sq, y_prime_sq)
    P_yy = np.matmul(curv_part_sq, x_prime_sq)
    P_xy = np.matmul(curv_part_sq, x_prime_y_prime)

    # ------------------------------------------------------------------------------------------------------------------
    # CONFIGURAR LAS MATRICES FINALES PARA EL SOLUCIONADOR -------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    T_nx = np.matmul(T_c, M_x)
    T_ny = np.matmul(T_c, M_y)

    H_x = np.matmul(T_nx.T, np.matmul(P_xx, T_nx))
    H_xy = np.matmul(T_ny.T, np.matmul(P_xy, T_nx))
    H_y = np.matmul(T_ny.T, np.matmul(P_yy, T_ny))
    H = H_x + H_xy + H_y
    H = (H + H.T) / 2   # hacer H simétrico

    f_x = 2 * np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xx, T_nx))
    f_xy = np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xy, T_ny)) \
           + np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_xy, T_nx))
    f_y = 2 * np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_yy, T_ny))
    f = f_x + f_xy + f_y
    f = np.squeeze(f)

    # ------------------------------------------------------------------------------------------------------------------
    # RESTRICCIONES KAPPA ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    Q_x = np.matmul(curv_part, y_prime)
    Q_y = np.matmul(curv_part, x_prime)

    # esta parte se multiplica por alfa dentro de la optimización (parte variable)
    E_kappa = np.matmul(Q_y, T_ny) - np.matmul(Q_x, T_nx)

    # parte de curvatura original (parte estática)
    k_kappa_ref = np.matmul(Q_y, np.matmul(T_c, q_y)) - np.matmul(Q_x, np.matmul(T_c, q_x))

    con_ge = np.ones((no_points, 1)) * kappa_bound - k_kappa_ref
    con_le = -(np.ones((no_points, 1)) * -kappa_bound - k_kappa_ref)  # multiplied by -1 as only LE conditions are poss.
    con_stack = np.append(con_ge, con_le)

    # ------------------------------------------------------------------------------------------------------------------
    # LLAMAR ALGORITMO DE PROGRAMACIÓN CUADRÁTICA ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    """
    descripción de la interfaz quadprog extraída de 
    https://github.com/stephane-caron/qpsolvers/blob/master/qpsolvers/quadprog_.py

    Resolver un programa cuadrático definido como:

        minimizar
            (1/2) * alpha.T * H * alpha + f.T * alpha

        sujeto a
            G * alpha <= h
            A * alpha == b

    usando quadprog <https://pypi.python.org/pypi/quadprog/>.

    Parametros
    ----------
    H : numpy.array
        Matriz simétrica de costes cuadráticos.
    f : numpy.array
        Vector de costes cuadráticos.
    G : numpy.array
        Matriz de restricciones de desigualdad lineal.
    h : numpy.array
        Vector de restricciones de desigualdad lineal.
    A : numpy.array, optional
        Matriz de restricciones de igualdad lineal.
    b : numpy.array, optional
        Vector de restricciones de igualdad lineal.
    initvals : numpy.array, optional
        Vector de suposición de arranque en caliente (no se utiliza).

    Returns
    -------
    alpha : numpy.array
            Solución a la QP, si se encuentra, de lo contrario ``None``.

    Nota
    ----
    El solucionador quadprog sólo tiene en cuenta las entradas inferiores de `H`, por lo que función de coste incorrecta 
    si se proporciona una matriz no simétrica.
    """

    # calcular la desviación permitida de la línea de referencia
    dev_max_right = reftrack[:, 2] - w_veh / 2
    dev_max_left = reftrack[:, 3] - w_veh / 2

    # limitar la trayectoria resultante a la línea de referencia en los puntos inicial y final de las vías abiertas
    if not closed and fix_s:
        dev_max_left[0] = 0.05
        dev_max_right[0] = 0.05

    if not closed and fix_e:
        dev_max_left[-1] = 0.05
        dev_max_right[-1] = 0.05

    # compruebe que queda espacio entre la desviación máxima izquierda y derecha (¡ambas también pueden ser negativas!)
    if np.any(-dev_max_right > dev_max_left) or np.any(-dev_max_left > dev_max_right):
        raise RuntimeError("Problema no solucionable, la vía podría ser demasiado pequeña para circular con la "
                           "distancia de seguridad actual!")

    # considerar los límites de valor (-dev_max_left <= alpha <= dev_max_right)
    G = np.vstack((np.eye(no_points), -np.eye(no_points), E_kappa, -E_kappa))
    h = np.append(dev_max_right, dev_max_left)
    h = np.append(h, con_stack)

    # guardar hora de inicio
    t_start = time.perf_counter()

    # resolver problema (quadprog) -------------------------------------------------------------------------------------
    alpha_mincurv = quadprog.solve_qp(H, -f, -G.T, -h, 0)[0]

    # imprimir el tiempo de ejecución en consola
    if print_debug:
        print("Tiempo de ejecución solver opt_min_curv: " + "{:.3f}".format(time.perf_counter() - t_start) + "s")

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULAR EL ERROR DE CURVATURA -----------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calcular la curvatura una vez a partir de la linealización original y otra vez a partir de una nueva linealización
    # en torno a la solución
    q_x_tmp = q_x + np.matmul(M_x, np.expand_dims(alpha_mincurv, 1))
    q_y_tmp = q_y + np.matmul(M_y, np.expand_dims(alpha_mincurv, 1))

    x_prime_tmp = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_x_tmp)
    y_prime_tmp = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_y_tmp)

    x_prime_prime = np.squeeze(np.matmul(T_c, q_x) + np.matmul(T_nx, np.expand_dims(alpha_mincurv, 1)))
    y_prime_prime = np.squeeze(np.matmul(T_c, q_y) + np.matmul(T_ny, np.expand_dims(alpha_mincurv, 1)))

    curv_orig_lin = np.zeros(no_points)
    curv_sol_lin = np.zeros(no_points)

    for i in range(no_points):
        curv_orig_lin[i] = (x_prime[i, i] * y_prime_prime[i] - y_prime[i, i] * x_prime_prime[i]) \
                          / math.pow(math.pow(x_prime[i, i], 2) + math.pow(y_prime[i, i], 2), 1.5)
        curv_sol_lin[i] = (x_prime_tmp[i, i] * y_prime_prime[i] - y_prime_tmp[i, i] * x_prime_prime[i]) \
                           / math.pow(math.pow(x_prime_tmp[i, i], 2) + math.pow(y_prime_tmp[i, i], 2), 1.5)

    if plot_debug:
        plt.plot(curv_orig_lin)
        plt.plot(curv_sol_lin)
        plt.legend(("linealización original", "linealización basada en la solución"))
        plt.show()

    # calcular el error máximo de curvatura
    curv_error_max = np.amax(np.abs(curv_sol_lin - curv_orig_lin))

    return alpha_mincurv, curv_error_max


# testing --------------------------------------------------------------------------------------------------------------

def main():
    import numpy as np
    import pickle
    import os
    import sys
    import matplotlib.pyplot as plt

    sys.path.append(os.path.abspath("../.."))
    from spline.calc_splines import calc_splines

    # --- PARAMETROS ---
    CLOSED = True

    psi_s = -2.0
    psi_e = 2.0
    # kappa_bound = 0.08
    kappa_bound = 0.08
    w_veh = 1.0
    ancho_total = 4
    width = (ancho_total - w_veh) / 2 # Este sera ek

    file_path = '../../tracks/trayectoria_smoth_00.pickle'  # Pista ordenada con trayectoria central definida
    # Cargar los arrays desde el archivo pickle
    with open(file_path, 'rb') as file:
        loaded_dict = pickle.load(file)

    mid_path = loaded_dict['waypoints']
    print(mid_path.shape)
    azules = loaded_dict['azules']
    amarillos = loaded_dict['amarillos']
    reftrack = []

    if CLOSED:
        for act in mid_path:
            reftrack.append([act[0], act[1], width, width])
        reftrack = np.array(reftrack)
        coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=np.vstack((reftrack[:, 0:2], reftrack[0, 0:2])))
    else:
        num = 500
        for act in mid_path[:num]:
            reftrack.append([act[0], act[1], width, width])

        reftrack = np.array(reftrack)
        coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=reftrack[:, 0:2],
                                                           psi_s=psi_s,
                                                           psi_e=psi_e)

        # extend norm-vec to same size of ref track (quick fix for testing only)
        # normvec_norm = np.vstack((normvec_norm[0, :], normvec_norm))

    # --- CALCULATE MIN CURV ---
    alpha_mincurv, curv_error_max = optimizar_minima_curvatura(reftrack=reftrack, normvectors=normvec_norm, A=M,
                                                               kappa_bound=kappa_bound, w_veh=w_veh, print_debug=True,
                                                               plot_debug=True, closed=CLOSED, psi_s=psi_s,
                                                               psi_e=psi_e)  # Solo para cuando no esta cerrado

    # --- PLOT RESULTS ---
    path_result = reftrack[:, 0:2] + normvec_norm * np.expand_dims(alpha_mincurv, axis=1)
    path_result = np.vstack((path_result, path_result[0, :]))
    bound_superior = reftrack[:, 0:2] - normvec_norm * np.expand_dims(reftrack[:, 2], axis=1)
    bound_inferiror = reftrack[:, 0:2] + normvec_norm * np.expand_dims(reftrack[:, 3], axis=1)

    plt.plot(bound_superior[:,0], bound_superior[:,1], ':', color='purple', label = 'Limites')
    plt.plot([bound_superior[-1, 0], bound_superior[0, 0]], [bound_superior[-1, 1], bound_superior[0, 1]], ':',
             color='purple')
    plt.plot(bound_inferiror[:, 0], bound_inferiror[:, 1], ':', color='purple')
    plt.plot([bound_inferiror[-1, 0], bound_inferiror[0, 0]], [bound_inferiror[-1, 1], bound_inferiror[0, 1]], ':',
             color='purple')

    plt.plot(path_result[:, 0], path_result[:, 1], '-', color='orange', label='Optimización')
    # plt.plot([path_result[-1, 0], path_result[0, 0]], [path_result[-1, 1], path_result[0, 1]], '-', color='orange')

    plt.plot([azules[0, 0], amarillos[0, 0]], [azules[0, 1], amarillos[0, 1]], '-', color='red')

    plt.plot(reftrack[:, 0], reftrack[:, 1], ':', color='green', label = 'Referencia')
    plt.plot([reftrack[-1, 0], reftrack[0, 0]], [reftrack[-1, 1], reftrack[0, 1]], ':', color='green')

    plt.plot(azules[:, 0], azules[:, 1], '-', color='blue')
    plt.plot([azules[-1, 0], azules[0, 0]], [azules[-1, 1], azules[0, 1]], '-', color='blue')
    plt.plot(azules[:, 0], azules[:, 1], 'o', markerfacecolor='blue', markeredgecolor='black', markersize=7)

    plt.plot(amarillos[:, 0], amarillos[:, 1], '-', color='yellow')
    plt.plot([amarillos[-1, 0], amarillos[0, 0]], [amarillos[-1, 1], amarillos[0, 1]], '-', color='yellow')
    plt.plot(amarillos[:, 0], amarillos[:, 1], 'o', markerfacecolor='yellow', markeredgecolor='black', markersize=7)

    plt.axis('equal')
    plt.legend()
    plt.title('Optimización de mínima curvatura')
    plt.show()


if __name__ == "__main__":
    main()
