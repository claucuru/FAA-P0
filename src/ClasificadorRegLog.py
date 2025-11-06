from Clasificador import Clasificador
from collections import Counter
import numpy as np
from Datos import Datos
from math import sqrt, pi, exp




class ClasificadorRegLog(Clasificador):
    def __init__(self, eta: int, epocas: int):
        self.eta = eta
        self.w = None
        self.epocas = epocas

    def entrenamiento(self, datos: Datos):
        n_attrs = datos.datos.shape[0] - 1
        
        # Vector aleatorio wT
        self.w = [np.random.uniform(-0.5, 0.5) for _ in n_attrs]        

        for epoca in range(self.epocas):
            for fila in datos.datos.iterrows():
                sigmoide = 1 /(1 +  np.e ** - np.dot(self.w, fila[:-1]))
                self.w = self.w - ( self.eta * (sigmoide - fila[-1]) ) * fila[:-1]





    def clasifica(self, datos: Datos):
        pass
