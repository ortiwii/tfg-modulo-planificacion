import numpy as np

def local_planification(conos_azules: np.ndarray, conos_amarillos: np.ndarray,conos_naranjas: np.ndarray,
                        conos_narnajas_grandes: np.ndarray, track_build: bool, planificador: str = 'delaunay',
                        posicion: np.ndarray = np.array([0.00, 0.00])):

    print('EJECUTANDO PLANIFICADOR ', planificador)