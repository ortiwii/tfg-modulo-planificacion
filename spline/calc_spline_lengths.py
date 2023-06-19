import numpy as np
import math


def calc_spline_lengths(coeffs_x: np.ndarray,
                        coeffs_y: np.ndarray,
                        quickndirty: bool = False,
                        no_interp_points: int = 15) -> np.ndarray:
    """
    .. description::
    Calcular longitudes de splines de tercer orden definiendo coordenadas x e y mediante pasos intermedios.

    .. inputs::
    :param coeffs_x:            matriz de coeficientes de los splines x con tamaño (no_splines x 4).
    :type coeffs_x:             np.ndarray
    :param coeffs_y:            matriz de coeficientes de los splines y con tamaño (no_splines x 4).
    :type coeffs_y:             np.ndarray
    :param quickndirty:         El valor True devuelve longitudes basadas en la distancia entre el primer y el último
                                punto de la spline en lugar de utilizar la interpolación.
    :type quickndirty:          bool
    :param no_interp_points:    el cálculo de la longitud se realiza con el número de pasos de interpolación dado
    :type no_interp_points:     int

    .. outputs::
    :return spline_lengths:     longitud de cada segmento de spline.
    :rtype spline_lengths:      np.ndarray

    .. notes::
    len(coeffs_x) = len(coeffs_y) = len(spline_lengths)
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARACIONES ----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Comprobar inputs
    if coeffs_x.shape[0] != coeffs_y.shape[0]:
        raise RuntimeError("Las matrices de coeficientes deben tener la misma longitud !")

    if coeffs_x.size == 4 and coeffs_x.shape[0] == 4:
        coeffs_x = np.expand_dims(coeffs_x, 0)
        coeffs_y = np.expand_dims(coeffs_y, 0)

    # obtener el número de splines y crear un array de salida
    no_splines = coeffs_x.shape[0]
    spline_lengths = np.zeros(no_splines)

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULAR TAMAÑOS -------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if quickndirty:
        for i in range(no_splines):
            spline_lengths[i] = math.sqrt(math.pow(np.sum(coeffs_x[i]) - coeffs_x[i, 0], 2)
                                          + math.pow(np.sum(coeffs_y[i]) - coeffs_y[i, 0], 2))

    else:
        # recorre todas los splines y calcula las coordenadas intermedias
        t_steps = np.linspace(0.0, 1.0, no_interp_points)
        spl_coords = np.zeros((no_interp_points, 2))

        for i in range(no_splines):
            spl_coords[:, 0] = coeffs_x[i, 0] \
                               + coeffs_x[i, 1] * t_steps \
                               + coeffs_x[i, 2] * np.power(t_steps, 2) \
                               + coeffs_x[i, 3] * np.power(t_steps, 3)
            spl_coords[:, 1] = coeffs_y[i, 0] \
                               + coeffs_y[i, 1] * t_steps \
                               + coeffs_y[i, 2] * np.power(t_steps, 2) \
                               + coeffs_y[i, 3] * np.power(t_steps, 3)

            spline_lengths[i] = np.sum(np.sqrt(np.sum(np.power(np.diff(spl_coords, axis=0), 2), axis=1)))

    return spline_lengths
