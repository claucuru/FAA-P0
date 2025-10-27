from abc import ABCMeta,abstractmethod
from collections import Counter
from enum import Enum
import numpy as np
from Datos import Datos
from EstrategiaParticionado import EstrategiaParticionado
from cmath import sqrt, pi, exp


class Clasificador:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto. Crea el modelo a partir de los datos de entrenamiento
  # datosTrain: matriz numpy o dataframe con los datos de entrenamiento
  # nominalAtributos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables
  def entrenamiento(self, datos: Datos):
    pass
  
  
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto. Devuelve un numpy array con las predicciones
  # datosTest: matriz numpy o dataframe con los datos de validaci�n
  # nominalAtributos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables
  # devuelve un numpy array o vector con las predicciones (clase estimada para cada fila de test)
  def clasifica(self, datos: Datos):
    pass
  
  
  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  # TODO: implementar
  # datos: los datos de test reales
  # pred: la lista de predicciones de clase (de los datos de test)
  def error(self,datos: Datos, pred):
    # Aqui se compara la prediccion (pred) con las clases reales de test (datos) y se calcula el error
    # devuelve el error
	pass
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # particionado: un objeto EstrategiaParticionado (Simple o Cruzada)
  # dataset: un objeto Datos
  # clasificador: un objeto de una subclase de Clasificador (ClasificadorNB...)
  # TODO: implementar esta funcion
  def validacion(self, particionado: EstrategiaParticionado, dataset: Datos, seed=None):
       
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta n-folds entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.
    # devuelve el vector con los errores por cada partici�n
    
    # pasos
    # crear particiones
    # inicializar vector de errores
    # for cada partici�n
    #     obtener datos de train
    #     obtener datos de test
    #     entrenar sobre los datos de train
    #     obtener prediciones de los datos de test (llamando a clasifica)
    #     a�adir error de la partici�n al vector de errores
	pass  

####################################################################################################################################

  


        
    

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
        if  self.datosTrain is None:
            return

        self.datosTrain.estandarizarDatos(True, True)
        datos.estandarizarDatos(True, True)

        predicciones = np.ndarray(datos.datos.shape[0])

        print("\n\nDATOS TEST:\n", "-"*20)
        print(datos.datos.iloc[:,:-1])

        # Hacemos la prediccion de clase para cada dato que queremos clasificar.
        # Cogemos todos los atributos menos la clase
        for sampleTest in datos.datos.iloc[:,:]:
          # Calculamos las distancias de la muestra a cada vecino
          distancias = []

          for i, sampleTrain in enumerate(self.datosTrain.datos.iloc[:,:-1]):
            dist = 0

            if self.distanceMetric == DistanceMetricKNN.EUCLIDES:
              dist = np.sum( (sampleTest - sampleTrain) ** 2 )
            elif self.distanceMetric == DistanceMetricKNN.MANHATTAN:
              pass

            print(f"Distancia( {sampleTest} , {sampleTrain}) =", dist)

            distancias.append( (dist, i) )

          # Ordenamos la lista de vecinos
          distancias.sort(key=lambda e1, e2 : e1[0] - e2[0])
          print("Distancias a vecinos:")
          print(distancias)

          # Cogemos los k vecinos mas cercanos
          cercanos = distancias[0:self.k]

          counter = defaultdict(int)
          for vecino in cercanos:
             clase = self.datosTrain[i][-1]
             counter[clase] += 1

          predicciones[i] = Counter(counter).most_common(1)[0][1]

          print("Clases de los K vecinos:")
          print(counter)

        return predicciones

