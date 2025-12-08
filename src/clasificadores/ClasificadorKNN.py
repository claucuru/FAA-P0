from collections import Counter
import time
import numpy as np
from .Clasificador import Clasificador
from .DistanceMetric import DistanceMetric
from Datos import Datos



class ClasificadorKNN(Clasificador):
    def __init__(self, k: int, distanceMetric: DistanceMetric):
        self.k = k
        self.distanceMetric = distanceMetric
        self._datosTrain = None


    def entrenamiento(self, datos: Datos):
        self._datosTrain = datos


    def clasifica(self, datos: Datos):
        if self._datosTrain is None:
            raise Exception("No se puede clasificar sin un entrenamiento previo")

        dataTrain = self._datosTrain.datos
        dataTest = datos.datos

        # Asumimos que la ultima columna corresponde a la clase
        n_attrs = dataTrain.shape[1] - 1

        # Array de predicciones de las clases para cada muestra del conjunto de test
        predicciones = np.ndarray(dataTest.shape[0])

        # Predecimos la clase para cada muestra por clasificar a partir de las
        # clases de los K vecinos mas cercanos.
        for idxTest, (_, sampleTest) in enumerate(dataTest.iloc[:,:n_attrs].iterrows()):

            match self.distanceMetric:
                case DistanceMetric.EUCLIDES:
                    distancias = np.sum(np.square(dataTrain.iloc[:,:n_attrs] - sampleTest), axis=1)
                case DistanceMetric.MANHATTAN:
                    distancias = np.sum(np.abs(dataTrain.iloc[:,:n_attrs] - sampleTest), axis=1)
                case _:
                    raise ValueError("Unsupported Distance Metric")

            # Obtenemos la lista de indices que ordenarian las distancias 
            # WARNING: No resuelve empates
            k_idx = np.argsort(distancias)[0:self.k]
            k_clases = self._datosTrain.datos.iloc[k_idx, -1]

            # Extraemos la clase mas repetida entre los vecinos
            # WARNING: No se valoran empates entre clases
            predicciones[idxTest] = Counter(k_clases).most_common(1)[0][0]

        return predicciones
