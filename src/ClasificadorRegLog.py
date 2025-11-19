from Clasificador import Clasificador
from collections import Counter
import numpy as np
from Datos import Datos
from math import sqrt, pi, exp


class ClasificadorRegLog(Clasificador):
    def __init__(self, eta: int, epocas: int, seed: int = 33):
        self.eta = eta
        self.w = None
        self.epocas = epocas
        self.seed = 33
        self._n_attrs = -1

    def entrenamiento(self, datos: Datos):
        self._n_attrs = datos.datos.shape[1] - 1

        # Vector aleatorio wT de tamano n_atributos + 1. Al vector de atributos
        # (w1, ..., wd) de cada muestra de d atributos se anade una dimension
        # fija con valor x0 = 1, de forma que el vector quedara como (w1, ..., wd).
        np.random.seed(self.seed)
        self.w = np.random.uniform(-0.5, 0.5, size=(self._n_attrs + 1))
        #self.w[0] = 1

        for epoch in range(self.epocas):
            print(f"{epoch=}", end="  ")

            sum = 0
            for i, fila in datos.datos.iterrows():
                # print(i)
                sum += 1
                # print(self.w)
                #print(fila)
                fila = fila.to_numpy()
                attrs = fila[:-1]
                clase = fila[-1]

                #print(f"{attrs.shape=}, {clase=}")

                sigmoide = 1 / (1 +  np.e ** ( -(self.w[0] + np.dot(self.w[1:], attrs)) ))
                gradiente = self.eta * (sigmoide - clase)

                #print(sigmoide, gradiente[:3])
                self.w[0] = self.w[0] - gradiente
                self.w[1:] = self.w[1:] - (gradiente * attrs)

            print(sum, self.w[:5])

    def clasifica(self, datos: Datos):
        predicciones = np.empty(datos.datos.shape[0])

        sum = 0
        for i, (_, dato) in enumerate(datos.datos.iterrows()):
            attrs = dato.to_numpy()[:self._n_attrs]
            sum += 1

            sigmoide = 1 / (1 +  np.e ** ( -(self.w[0] + np.dot(self.w[1:], attrs)) ))

            if sigmoide < 0.5:
                predicciones[i] = 0
            else:
                predicciones[i] = 1

            #print(f"Dato {i}: esperado={dato.to_numpy()[-1]}, resultado={predicciones[i]}")


        #print("Num Test: ", sum)
        #print(self.w)

        return predicciones
