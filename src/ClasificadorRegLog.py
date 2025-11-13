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
        self._n_attrs = -1

    def entrenamiento(self, datos: Datos):
        self._n_attrs = datos.datos.shape[1] - 1

        # Vector aleatorio wT
        self.w = [np.random.uniform(-0.5, 0.5) for _ in range(self._n_attrs)]        

        for epoch in range(self.epocas):
            print(f"{epoch=}", end="  ")

            sum = 0
            for i, fila in datos.datos.iterrows():
                # print(i)
                sum += 1
                # print(self.w)
                #print(fila)
                f = fila.to_numpy()

                sigmoide = 1 / (1 +  np.e ** (- np.dot(self.w, f[:-1])) )
                self.w = self.w - ( self.eta * (sigmoide - f[-1]) ) * f[:-1]

            print(sum)

    def clasifica(self, datos: Datos):
        predicciones = np.empty(datos.datos.shape[0])

        sum = 0
        for i, (_, dato) in enumerate(datos.datos.iterrows()):
            f = dato.to_numpy()[:self._n_attrs]
            sum += 1
            

            sigmoide = 1 / (1 +  np.e ** (- np.dot(self.w, f)) )

            if sigmoide < 0.5:
                predicciones[i] = 0
            else:
                predicciones[i] = 1

        print("Num Test: ", sum)
        print(self.w)

        return predicciones
