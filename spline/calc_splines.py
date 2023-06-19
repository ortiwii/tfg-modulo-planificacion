import numpy as np
import math


def calc_splines(path: np.ndarray,
                 el_lengths: np.ndarray = None,
                 psi_s: float = None,
                 psi_e: float = None,
                 use_dist_scaling: bool = True) -> tuple:
    """
    .. description::
    Resolver splines cúbicos continuos de curvatura (parámetro de spline t) entre puntos i dados (splines evaluados
    en t = 0 y t = 1). Las splines deben establecerse por separado para las coordenadas x e y.
    Ecuaciones spline:
    P_{x,y}(t)   =  a_3 * t³ +  a_2 * t² + a_1 * t + a_0
    P_{x,y}'(t)  = 3a_3 * t² + 2a_2 * t  + a_1
    P_{x,y}''(t) = 6a_3 * t  + 2a_2

    a * {x; y} = {b_x; b_y}

    .. inputs::
    :param path:                coordenadas x e y como base para la construcción de la spline (cerrada o no cerrada).
                                Si la trayectoria se proporciona sin cerrar, ¡se requieren las rúbricas psi_s y psi_e!
    :type path:                 np.ndarray
    :param el_lengths:          distancias entre puntos de trayectoria (cerrada o no cerrada). La entrada es opcional.
                                Las distancias son necesarias para escalar los valores de rumbo y curvatura. Se calculan
                                utilizando distancias euclidianas si se requieren pero no se suministran.
    :type el_lengths:           np.ndarray
    :param psi_s:               orientación del punto {inicial, final}.
    :type psi_s:                float
    :param psi_e:               orientación del punto {inicial, final}..
    :type psi_e:                float
    :param use_dist_scaling:    flag bool para indicar si se debe realizar el escalado de rumbo y curvatura.
                                Esto debe hacerse si las distancias entre los puntos de la trayectoria no son iguales.
    :type use_dist_scaling:     bool

    .. outputs::
    :return x_coeff:            coeficientes spline del componente x.
    :rtype x_coeff:             np.ndarray
    :return y_coeff:            coeficientes spline del componente y.
    :rtype y_coeff:             np.ndarray
    :return M:                  coeficientes LES.
    :rtype M:                   np.ndarray
    :return normvec_normalized: vectores normales normalizados [x, y].
    :rtype normvec_normalized:  np.ndarray

    .. notes::
    Las entradas path y el_lengths pueden ser cerradas o no cerradas, ¡pero deben ser coherentes! La función detecta
    automáticamente si el camino se insertó cerrado.

    Las matrices de coeficientes tienen la forma a_0i, a_1i * t, a_2i * t^2, a_3i * t^3.
    """

    # check if path is closed
    if np.all(np.isclose(path[0], path[-1])) and psi_s is None:
        closed = True
    else:
        closed = False

    # comprobar inputs
    if not closed and (psi_s is None or psi_e is None):
        raise RuntimeError("Las orientaciones {inicio,fin} deben indicarse para el cálculo de splines no cerrados!")

    if el_lengths is not None and path.shape[0] != el_lengths.size + 1:
        raise RuntimeError("La entrada el_lengths debe ser un elemento menor que la entrada path!")

    # si no se proporcionan las distancias entre las coordenadas de la ruta pero se requieren, calcular las distancias
    # euclidianas como el_lengths
    if use_dist_scaling and el_lengths is None:
        el_lengths = np.sqrt(np.sum(np.power(np.diff(path, axis=0), 2), axis=1))
    elif el_lengths is not None:
        el_lengths = np.copy(el_lengths)

    # si está cerrado y use_dist_scaling activo añada la longitud del elemento para obtener elementos superpuestos para
    # el correcto escalado posterior del último elemento
    if use_dist_scaling and closed:
        el_lengths = np.append(el_lengths, el_lengths[0])

    # obtener el numero de splines
    no_splines = path.shape[0] - 1

    # calcular los factores de escala entre cada par de splines
    if use_dist_scaling:
        scaling = el_lengths[:-1] / el_lengths[1:]
    else:
        scaling = np.ones(no_splines - 1)

    # ------------------------------------------------------------------------------------------------------------------
    # DEFINIR SISTEMA DE ECUACIONES LINEALES ---------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    M = np.zeros((no_splines * 4, no_splines * 4))
    b_x = np.zeros((no_splines * 4, 1))
    b_y = np.zeros((no_splines * 4, 1))

    # crear plantilla para M entradas al array
    # columna 1: el inicio de la spline actual debe situarse en el punto actual (t = 0)
    # columna 2: el final de la spline actual debe situarse en el siguiente punto (t = 1)
    # columna 3: el rumbo al final de la spline actual debe ser igual al rumbo al principio de la siguiente spline
    #            (t = 1 y t = 0)
    # columna 4: la curvatura al final de la spline actual debe ser igual a la curvatura al principio de la siguiente
    #            spline (t = 1 y t = 0)

    template_M = np.array(                          # punto actual                | siguiente punto         | limites
                [[1,  0,  0,  0,  0,  0,  0,  0],   # a_0i                                                  = {x,y}_i
                 [1,  1,  1,  1,  0,  0,  0,  0],   # a_0i + a_1i +  a_2i +  a_3i                           = {x,y}_i+1
                 [0,  1,  2,  3,  0, -1,  0,  0],   # _      a_1i + 2a_2i + 3a_3i      - a_1i+1             = 0
                 [0,  0,  2,  6,  0,  0, -2,  0]])  # _             2a_2i + 6a_3i               - 2a_2i+1   = 0

    for i in range(no_splines):
        j = i * 4

        if i < no_splines - 1:
            M[j: j + 4, j: j + 8] = template_M

            M[j + 2, j + 5] *= scaling[i]
            M[j + 3, j + 6] *= math.pow(scaling[i], 2)

        else:
            # sin límites de curvatura y rumbo en el último elemento (se gestionan posteriormente)
            M[j: j + 2, j: j + 4] = [[1,  0,  0,  0],
                                     [1,  1,  1,  1]]

        b_x[j: j + 2] = [[path[i,     0]],
                         [path[i + 1, 0]]]
        b_y[j: j + 2] = [[path[i,     1]],
                         [path[i + 1, 1]]]

    # ------------------------------------------------------------------------------------------------------------------
    # ESTABLECER CONDICIONES DE CONTORNO PARA EL ÚLTIMO Y EL PRIMER PUNTO ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if not closed:
        # si la trayectoria no es cerrada, queremos fijar el rumbo en el punto inicial y final de la trayectoria
        # (la curvatura no puede determinarse en este caso) -> fijar las condiciones de contorno del rumbo

        # direccion punto inicial
        M[-2, 1] = 1

        if el_lengths is None:
            el_length_s = 1.0
        else:
            el_length_s = el_lengths[0]

        b_x[-2] = math.cos(psi_s + math.pi / 2) * el_length_s
        b_y[-2] = math.sin(psi_s + math.pi / 2) * el_length_s

        # direccion punto final
        M[-1, -4:] = [0, 1, 2, 3]

        if el_lengths is None:
            el_length_e = 1.0
        else:
            el_length_e = el_lengths[-1]

        b_x[-1] = math.cos(psi_e + math.pi / 2) * el_length_e
        b_y[-1] = math.sin(psi_e + math.pi / 2) * el_length_e

    else:
        M[-2, 1] = scaling[-1]
        M[-2, -3:] = [-1, -2, -3]

        M[-1, 2] = 2 * math.pow(scaling[-1], 2)
        M[-1, -2:] = [-2, -6]

    # ------------------------------------------------------------------------------------------------------------------
    # RESOLVER ---------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    x_les = np.squeeze(np.linalg.solve(M, b_x))  # squeeze elimina las entradas unidimensionales
    y_les = np.squeeze(np.linalg.solve(M, b_y))

    coeffs_x = np.reshape(x_les, (no_splines, 4))
    coeffs_y = np.reshape(y_les, (no_splines, 4))

    # obtener vector normal (aquí se utiliza detrás en lugar de delante por coherencia con otras funciones)
    # (el segundo coeficiente de los splines cúbicos es relevante para el encabezamiento)
    normvec = np.stack((coeffs_y[:, 1], -coeffs_x[:, 1]), axis=1)

    # normalizar vectores normales
    norm_factors = 1.0 / np.sqrt(np.sum(np.power(normvec, 2), axis=1))
    normvec_normalized = np.expand_dims(norm_factors, axis=1) * normvec

    return coeffs_x, coeffs_y, M, normvec_normalized
