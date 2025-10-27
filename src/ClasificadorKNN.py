from enum import Enum
from collections import Counter, defaultdict
import numpy as np
from Clasificador import Clasificador
from Datos import Datos


class DistanceMetricKNN(Enum):
    EUCLIDES = 1
    MANHATTAN = 2


class KNN(Clasificador):
    def __init__(self, k: int, distanceMetric: DistanceMetricKNN):
        self.k = k
        self.distanceMetric = distanceMetric
        self.datosTrain = None


    def entrenamiento(self, datos: Datos):
        """
        Se deben estandarizar fuera
        """
        self.datosTrain = datos


    def clasifica(self, datos: Datos):
        if self.datosTrain is None:
            return

        n_attrs = self.datosTrain.datos.shape[1] - 1


        # Se deben estandarizar los datos antes de aplicar KNN
        dataTrain = self.datosTrain.estandarizarDatos(True, True)
        dataTest  = datos.estandarizarDatos(True, True)

        # Array de predicciones de las clases para cada muestra
        # del conjunto de test
        predicciones = np.ndarray(dataTest.shape[0])

        print("\n\nDATOS TEST:\n", "-"*20)
        print(dataTest.iloc[:,:n_attrs], end="\n\n")

        # Hacemos la prediccion de clase para cada dato que queremos clasificar.
        # Cogemos todos los atributos menos la clase
        for idxTest, sampleTest in dataTest.iloc[:,:n_attrs].iterrows():
          # Calculamos las distancias de la muestra a cada vecino
          distancias = []

          for idxTrain, sampleTrain in dataTrain.iloc[:,:n_attrs].iterrows():
            dist = 0

            print("==DISTANCIA====================\n")
            print("TRAIN:")
            print(sampleTrain)
            print("\nTEST:")
            print(sampleTest)

            if self.distanceMetric == DistanceMetricKNN.EUCLIDES:
              dist = np.sum( (sampleTest - sampleTrain) ** 2 )
            elif self.distanceMetric == DistanceMetricKNN.MANHATTAN:
              pass

            print(f"\nDISTANCE={dist}")

            distancias.append( (dist, idxTrain) )

          # Ordenamos la lista de vecinos
          distancias.sort(key=lambda e1 : e1[0])
          print("Distancias a vecinos:")
          print(distancias)

          # Cogemos los k vecinos mas cercanos
          cercanos = distancias[0:self.k]

          counter = defaultdict(int)
          for dist, idxTrain in cercanos:
             neighbour_class = self.datosTrain.datos.iat[idxTrain, -1]
             counter[neighbour_class] += 1

          # Tomamos la clase mas comun entre los vecinos mas
          # cercanos a la muestra de test
          most_freq_class, freq = Counter(counter).most_common(1)[0]

          predicciones[idxTest] = most_freq_class

        return predicciones
