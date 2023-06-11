import pickle
import pandas as pd
import numpy as np

def load_pickle_map(file_path):
    with open(file_path, "rb") as fp:
        return pickle.load(fp)
    return


def load_csv_path(file_path):
    return pd.read_csv(file_path)


def save_pickle_map(map, file_path):
    try:
        with open(file_path, 'wb') as archivo:
            pickle.dump(map, archivo)

        print("Mapa guardado en en " + file_path)
    except:
        print("No se ha podido guardar el mapa correctamente en " + file_path)

def calcular_tramos_por_distancia(trayectoria, num_tramos):
    # TODO: Mejorar resultados

    # Obtener las coordenadas x e y de la trayectoria
    x = trayectoria[:, 0]
    y = trayectoria[:, 1]

    # Calcular las diferencias en las coordenadas x e y
    dx = np.diff(x)
    dy = np.diff(y)

    # Calcular la distancia entre cada par de puntos consecutivos
    distancias = np.sqrt(dx**2 + dy**2)

    # Calcular la distancia total sumando todas las distancias parciales
    distancia_total = np.sum(distancias)

    # Calcular la distancia promedio por tramo
    distancia_promedio = distancia_total / num_tramos

    # Inicializar variables
    inicio_tramo = 0
    fin_tramo = 0
    distancia_actual = 0
    tramos = []

    # Iterar sobre las distancias y encontrar los índices de inicio y fin de cada tramo
    for i, distancia in enumerate(distancias):
        distancia_actual += distancia
        if distancia_actual >= distancia_promedio:
            fin_tramo = i + 1 # Sumar 1 para incluir el punto final en el tramo
            tramos.append((inicio_tramo, fin_tramo))
            distancia_actual = 0
            inicio_tramo = fin_tramo

    return np.array(tramos)