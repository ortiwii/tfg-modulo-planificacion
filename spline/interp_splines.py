import numpy as np
import math
from calc_spline_lengths import calc_spline_lengths


def interp_splines(coeffs_x: np.ndarray,
                   coeffs_y: np.ndarray,
                   spline_lengths: np.ndarray = None,
                   incl_last_point: bool = False,
                   stepsize_approx: float = None,
                   stepnum_fixed: list = None) -> tuple:
    """
    .. description::
    Interpola puntos en una o más splines de tercer orden. El último punto (es decir, t = 1,0) se puede incluir si la
    opción se establece en consecuencia (se debe evitar para un raceline cerrado en la mayoría de los casos).
    El algoritmo mantiene stepize_approx lo mejor posible.

    .. inputs::
    :param coeffs_x:        matriz de coeficientes de los splines x con tamaño (no_splines x 4).
    :type coeffs_x:         np.ndarray
    :param coeffs_y:        matriz de coeficientes de los splines y con tamaño (no_splines x 4).
    :type coeffs_y:         np.ndarray
    :param spline_lengths:  matriz que contiene las longitudes de las splines insertadas con tamaño (no_splines x 1).
    :type spline_lengths:   np.ndarray
    :param incl_last_point: flag para establecer si el último punto debe mantenerse o eliminarse antes de la devolución.
    :type incl_last_point:  bool
    :param stepsize_approx: tamaño deseado de los puntos después de la interpolación.                   \\ Dar solo uno
    :type stepsize_approx:  float
    :param stepnum_fixed:   devolver un número fijo de coordenadas por spline, lista de longitud no_splines. \\ uno de los dos!
    :type stepnum_fixed:    list

    .. outputs::
    :return path_interp:    puntos de trayectoria interpolados.
    :rtype path_interp:     np.ndarray
    :return spline_inds:    contiene los índices de las splines que contienen los puntos interpolados.
    :rtype spline_inds:     np.ndarray
    :return t_values:       contiene los valores relativos de las coordenadas splines (t) de cada punto de las splines.
    :rtype t_values:        np.ndarray
    :return dists_interp:   distancia total hasta cada punto de interpolación.
    :rtype dists_interp:    np.ndarray

    .. notes::
    len(coeffs_x) = len(coeffs_y) = len(spline_lengths)

    len(path_interp = len(spline_inds) = len(t_values) = len(dists_interp)
    """

    # ------------------------------------------------------------------------------------------------------------------
    # CHECKS DE ENTRADA ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Compbar tamaños
    if coeffs_x.shape[0] != coeffs_y.shape[0]:
        raise RuntimeError("Coefficient matrices must have the same length!")

    if spline_lengths is not None and coeffs_x.shape[0] != spline_lengths.size:
        raise RuntimeError("coeffs_x/y and spline_lengths must have the same length!")

    # comprueba si coeffs_x y coeffs_y tienen exactamente dos dimensiones y genera un error en caso contrario
    if not (coeffs_x.ndim == 2 and coeffs_y.ndim == 2):
        raise RuntimeError("Coefficient matrices do not have two dimensions!")

    # comprobar si la especificación del tamaño del paso es válida
    if (stepsize_approx is None and stepnum_fixed is None) \
            or (stepsize_approx is not None and stepnum_fixed is not None):
        raise RuntimeError("Provide one of 'stepsize_approx' and 'stepnum_fixed' and set the other to 'None'!")

    if stepnum_fixed is not None and len(stepnum_fixed) != coeffs_x.shape[0]:
        raise RuntimeError("The provided list 'stepnum_fixed' must hold an entry for every spline!")

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULAR EL NÚMERO DE PUNTOS DE INTERPOLACIÓN Y LAS DISTANCIAS CORRESPONDIENTES ----------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if stepsize_approx is not None:
        # obtener la distancia total hasta el final de cada spline (es decir, las distancias acumuladas)
        if spline_lengths is None:
            spline_lengths = calc_spline_lengths(coeffs_x=coeffs_x,coeffs_y=coeffs_y,quickndirty=False)

        dists_cum = np.cumsum(spline_lengths)

        # calcular el número de puntos y distancias de interpolación (+1 porque el último punto se incluye al principio)
        no_interp_points = math.ceil(dists_cum[-1] / stepsize_approx) + 1
        dists_interp = np.linspace(0.0, dists_cum[-1], no_interp_points)

    else:
        # obtener el número total de puntos a muestrear (restar los puntos solapados)
        no_interp_points = sum(stepnum_fixed) - (len(stepnum_fixed) - 1)
        dists_interp = None

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULAR LOS PASOS INTERMEDIOS -------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # crear arrays para guardar los valores
    path_interp = np.zeros((no_interp_points, 2))           # array de coordenadas de la línea de carrera (x, y)
    spline_inds = np.zeros(no_interp_points, dtype=int)  # guardar el índice de spline al que pertenece un punto
    t_values = np.zeros(no_interp_points)                   # guardar valores t

    if stepsize_approx is not None:

        # --------------------------------------------------------------------------------------------------------------
        # APROX. STEP SIZE IGUAL A LO LARGO DE LA TRAYECTORIA DE SPLINES ADYACENTES -------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # recorrer todos los elementos y crear pasos con stepize_approx
        for i in range(no_interp_points - 1):
            # encontrar la spline que alberga el punto de interpolación actual
            j = np.argmax(dists_interp[i] < dists_cum)
            spline_inds[i] = j

            # obtener el valor t de la spline en función del avance dentro del elemento actual
            if j > 0:
                t_values[i] = (dists_interp[i] - dists_cum[j - 1]) / spline_lengths[j]
            else:
                if spline_lengths.ndim == 0:
                    t_values[i] = dists_interp[i] / spline_lengths
                else:
                    t_values[i] = dists_interp[i] / spline_lengths[0]

            # calcular coordenadas
            path_interp[i, 0] = coeffs_x[j, 0] \
                                + coeffs_x[j, 1] * t_values[i]\
                                + coeffs_x[j, 2] * math.pow(t_values[i], 2) \
                                + coeffs_x[j, 3] * math.pow(t_values[i], 3)

            path_interp[i, 1] = coeffs_y[j, 0]\
                                + coeffs_y[j, 1] * t_values[i]\
                                + coeffs_y[j, 2] * math.pow(t_values[i], 2) \
                                + coeffs_y[j, 3] * math.pow(t_values[i], 3)

    else:

        # --------------------------------------------------------------------------------------------------------------
        # STEP SIZE FIJO PARA CADA SEGMENTO DE SPLINE ---------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        j = 0

        for i in range(len(stepnum_fixed)):
            # omitir el último punto excepto el último segmento
            if i < len(stepnum_fixed) - 1:
                t_values[j:(j + stepnum_fixed[i] - 1)] = np.linspace(0, 1, stepnum_fixed[i])[:-1]
                spline_inds[j:(j + stepnum_fixed[i] - 1)] = i
                j += stepnum_fixed[i] - 1

            else:
                t_values[j:(j + stepnum_fixed[i])] = np.linspace(0, 1, stepnum_fixed[i])
                spline_inds[j:(j + stepnum_fixed[i])] = i
                j += stepnum_fixed[i]

        t_set = np.column_stack((np.ones(no_interp_points), t_values, np.power(t_values, 2), np.power(t_values, 3)))

        # eliminar muestras superpuestas
        n_samples = np.array(stepnum_fixed)
        n_samples[:-1] -= 1

        path_interp[:, 0] = np.sum(np.multiply(np.repeat(coeffs_x, n_samples, axis=0), t_set), axis=1)
        path_interp[:, 1] = np.sum(np.multiply(np.repeat(coeffs_y, n_samples, axis=0), t_set), axis=1)

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULAR EL ÚLTIMO PUNTO SI ES NECESARIO (t = 1,0) ---------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if incl_last_point:
        path_interp[-1, 0] = np.sum(coeffs_x[-1])
        path_interp[-1, 1] = np.sum(coeffs_y[-1])
        spline_inds[-1] = coeffs_x.shape[0] - 1
        t_values[-1] = 1.0

    else:
        path_interp = path_interp[:-1]
        spline_inds = spline_inds[:-1]
        t_values = t_values[:-1]

        if dists_interp is not None:
            dists_interp = dists_interp[:-1]

    # NOTA: dists_interp es None, cuando se utiliza un tamaño de paso fijo
    return path_interp, spline_inds, t_values, dists_interp


