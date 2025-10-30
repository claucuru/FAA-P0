from collections import Counter, defaultdict
from enum import Enum
import time
import numpy as np
from Clasificador import Clasificador
from Datos import Datos
from Estandarizador import Estandarizador


class DistanceMetricKNN(Enum):
    EUCLIDES  = 1
    MANHATTAN = 2


class KNN(Clasificador):
    def __init__(self, k: int, distanceMetric: DistanceMetricKNN):
        self.k = k
        self.distanceMetric = distanceMetric
        self._datosTrain = None
        self.attrsNominalesCuantitativos = None


    def entrenamiento(self, datos: Datos):
        self._datosTrain = datos


    def clasifica(self, datos: Datos):
        if self._datosTrain is None:
            raise Exception("No se puede clasificar sin un entrenamiento previo")

        # Time stats
        t_dist = 0          # Tiempo para calcular distancias entre muestras
        t_vecinos = 0       # Tiempo para determinar vecinos

        # Se deben estandarizar los datos antes de aplicar KNN para evitar
        # sesgos y problemas con las escalas de los atributos
        estandarizador = Estandarizador(self._datosTrain, True, True, None)
        dataTrain = estandarizador.estandarizarDatos(self._datosTrain.datos)
        dataTest  = estandarizador.estandarizarDatos(datos.datos)

        #print(dataTest)

        # Asumimos que la ultima columna corresponde a la clase
        n_attrs = dataTrain.shape[1] - 1

        # Array de predicciones de las clases para cada muestra
        # del conjunto de test
        predicciones = np.ndarray(dataTest.shape[0])

        # Array de distancias de la muestra por clasificar a cada muestra del
        # conjunto de datos de entrenamiento.
        distancias = np.ndarray(shape=(dataTrain.shape[0]), dtype=np.float64)

        # Predecimos la clase para cada muestra por clasificar a partir de las
        # clases de los K vecinos mas cercanos.
        for idxTest, sampleTest in dataTest.iloc[:,:n_attrs].iterrows():
            
            t_start = time.time()
            
            for idxTrain, sampleTrain in dataTrain.iloc[:,:n_attrs].iterrows():
                match self.distanceMetric:
                    case DistanceMetricKNN.EUCLIDES:
                        dist = np.sqrt(np.sum(np.square( (sampleTest - sampleTrain) )))
                    case DistanceMetricKNN.MANHATTAN:
                        dist = np.sum( np.abs( (sampleTest - sampleTrain)) )

                distancias[idxTrain] = dist

            t_dist += time.time() - t_start

            t_start = time.time()

            # Obtenemos la lista de indices que ordenarian las distancias 
            # WARNING: No resuelve empates
            k_idx = np.argsort(distancias)[0:self.k]
            k_clases = self._datosTrain.datos.iloc[k_idx, -1]

            # Extraemos la clase mas repetida entre los vecinos
            # WARNING: No se valoran empates entre clases
            predicciones[idxTest] = Counter(k_clases).most_common(1)[0][0]

            t_vecinos += time.time() - t_start


        print("Tiempo Calculo Distancias (sec):", t_dist)
        print("Tiempo Calculo vecinos (sec):", t_vecinos)

        return predicciones
