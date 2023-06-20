import numpy as np

def global_path_optimization(conos_azules: np.ndarray, conos_amarillos: np.ndarray,conos_naranjas: np.ndarray,
                             conos_narnajas_grandes: np.ndarray, track_build: bool, optimizator: str = 'min_curv',
                             posicion: np.ndarray = np.array([0.00, 0.00])):

    print('EJECUTANDO OPTIMIZADOR ', optimizator)
