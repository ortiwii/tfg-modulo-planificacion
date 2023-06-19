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


def calcular_tramos_ambos_lados(azules: np.ndarray, amarillos: np.ndarray, num_tramos: int):
    nAzules = azules.shape[0]
    nAmarillos = amarillos.shape[0]

    if nAzules <= nAmarillos:  # Hay mas amarillos que azules
        menores = azules
        mayores = amarillos
    else:  # Hay mas azules que amarillos
        mayores = azules
        menores = amarillos

    xMenores = menores[:, 0]
    yMenores = menores[:, 1]

    # Calcular las diferencias en las coordenadas x e y
    dxMenores = np.diff(xMenores)
    dyMenores = np.diff(yMenores)

    # Calcular la distancia entre cada par de puntos consecutivos
    distanciasMenores = np.sqrt(dxMenores ** 2 + dyMenores ** 2)

    # Calcular la distancia total sumando todas las distancias parciales
    distancia_total_menores = np.sum(distanciasMenores)

    distancia_promedio = distancia_total_menores / num_tramos + 1
    distancia_acumulada = 0

    # Iterar sobre las distancias y encontrar los índices de inicio y fin de cada tramo
    tramos_menores = []
    tramos_mayores = []

    tramo_act = 1

    i_tramo_prev_menores = 0
    i_tramo_prev_mayores = 0

    for i, distancia in enumerate(distanciasMenores):
        distancia_acumulada += distancia
        # print(distancia_acumulada)
        if distancia_acumulada >= tramo_act * distancia_promedio:  # Es el siguiente tramo
            # print('----->   TRAMO', tramo_act)
            # print([i_tramo_prev, i - 1])
            tramos_menores.append([i_tramo_prev_menores, i - 1])
            i_mayor = buscar_punto_cercano(mayores, menores[i-1])
            tramos_mayores.append([i_tramo_prev_mayores, i_mayor])

            i_tramo_prev_menores = i - 1
            i_tramo_prev_mayores = i_mayor
            tramo_act += 1

    tramos_menores.append([i_tramo_prev_menores, 0])
    tramos_mayores.append([i_tramo_prev_mayores, 0])

    if nAzules <= nAmarillos:  # siempre hay que devolver primero azules y luego amarillos
        return tramos_menores, tramos_mayores
    else:
        return tramos_mayores, tramos_menores

def buscar_punto_cercano(puntos, referencia):
    i_min = 0
    dist_min = 100000  # m
    for i, act in enumerate(puntos):
        distancia = np.sqrt((referencia[0] - act[0]) ** 2 + (referencia[1] - act[1]) ** 2)
        if distancia < dist_min:
            i_min = i
            dist_min = distancia
    return i_min


def calcular_tramos_por_distancia(trayectoria, num_tramos):
    # TODO: Mejorar resultados

    # Obtener las coordenadas x e y de la trayectoria
    x = trayectoria[:, 0]
    y = trayectoria[:, 1]

    # Calcular las diferencias en las coordenadas x e y
    dx = np.diff(x)
    dy = np.diff(y)

    # Calcular la distancia entre cada par de puntos consecutivos
    distancias = np.sqrt(dx ** 2 + dy ** 2)

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
            fin_tramo = i + 1  # Sumar 1 para incluir el punto final en el tramo
            tramos.append((inicio_tramo, fin_tramo))
            distancia_actual = 0
            inicio_tramo = fin_tramo

    return np.array(tramos)


def calcular_tramos(trayectoria, num_tramos):
    # print(len(trayectoria))
    # print('NUMERO DE TRAMOS', num_tramos)
    # Obtener las coordenadas x e y de la trayectoria
    x = trayectoria[:, 0]
    y = trayectoria[:, 1]

    # Calcular las diferencias en las coordenadas x e y
    dx = np.diff(x)
    dy = np.diff(y)

    # Calcular la distancia entre cada par de puntos consecutivos
    distancias = np.sqrt(dx ** 2 + dy ** 2)

    # Calcular la distancia total sumando todas las distancias parciales
    distancia_total = np.sum(distancias)
    # print('TOTAL', distancia_total)
    # Calcular la distancia promedio por tramo
    distancia_promedio = distancia_total / num_tramos + 1
    # print(distancia_promedio)
    distancia_acumulada = 0
    distancia_previa = 0

    # Iterar sobre las distancias y encontrar los índices de inicio y fin de cada tramo
    tramos = []
    tramo_act = 1
    i_tramo_prev = 0

    for i, distancia in enumerate(distancias):
        distancia_acumulada += distancia
        # print(distancia_acumulada)
        if distancia_acumulada >= (tramo_act) * distancia_promedio:  # Es el siguiente tramo
            # print('----->   TRAMO', tramo_act)
            # print([i_tramo_prev, i - 1])
            tramos.append([i_tramo_prev, i - 1])
            i_tramo_prev = i - 1
            tramo_act += 1
    tramos.append([i_tramo_prev, 0])
    # print('----->   TRAMO', tramo_act)
    # print([i_tramo_prev, 0])
    return np.array(tramos)
