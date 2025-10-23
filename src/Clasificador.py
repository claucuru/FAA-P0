from abc import ABCMeta,abstractmethod
from collections import Counter
from enum import Enum
import numpy as np
from Datos import Datos
from EstrategiaParticionado import EstrategiaParticionado


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

class MultinomialNB(Clasificador):

  # TODO: esta funcion debe ser implementada en cada clasificador concreto. Crea el modelo a partir de los datos de entrenamiento
  # datosTrain: matriz numpy o dataframe con los datos de entrenamiento
  # nominalAtributos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables
  def entrenamiento(self, datos: Datos):
    datos.ndim
    #datos.shape[0] numero de filas del data set
    #datos.shape[1] número de atributos (+ clase)


    #Saco una lista con todos los valores de clase:
    lista_clases = datos.datos[:, -1]
    
    #Cuento los valores de cada clase
    n_c = Counter(lista_clases)
    num_tot_clase = {int(k): v for k, v in n_c.items()}

    #Sacar el valor de numero de veces que aparece la clase entre el numero total de filas (shape[0]) => PRIORI

    prioris = {}
    for num in num_tot_clase:
      prioris[num] = num/datos.datos.shape[0]
    
    #Saco las clases que hay
    clases = np.unique(lista_clases)
    
    # Diccionario con probabilidades condicionadas
    condicionales = {c: {} for c in clases}

    datos = datos.datos.shape[0]
    
    for clase in clases:
      #Sacamos las filas de atributos pertenecientes a esa clase:
      filas_clase = datos.datos[datos.datos[:, -1] == clase]
      n_filas = len(filas_clase)

      for i in range (datos):
         #Distintos valores que puede tener un atributo i en la clase c y las veces que aparece ese valor en esa clase
         valores, counts = np.unique(filas_clase[:, i], return_counts=True)
         
         # Obtenemos el número de valores posibles del atributo i
         k = len(np.unique(datos[:, i]))
         #Diccionario para atributo i en la clase c
         condicionales[clase][i] = {}
        

         for v in np.unique(datos[:,i]):
            # Recorremos todos los posibles valores del atributo y almacenamos el número de veces que aparece en la clase
            if v in valores:
               count_v = counts[valores == v][0]
            else:
              count_v = 0

            condicionales[clase][i][v] = (count_v + 1) / (n_c + k)

    self.modelo = {"prioris": prioris, "condicionales":condicionales}


  # datosTest: matriz numpy o dataframe con los datos de validaci�n
  # nominalAtributos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables
  # devuelve un numpy array o vector con las predicciones (clase estimada para cada fila de test)
  def clasifica(self, datos: Datos):
    predicciones_clases = []

    for atrb in datos.datos:
      probabilidades_clase = {}   
      # Recorremos todas las clases
      for clase in self.modelo["prioris"]:
        #Cogemos la prioridad a priori de la clase
        p = self.modelo["prioris"][clase]

        # Para todos los atributos menos la clase
        for i, valor in enumerate(datos.datos[atrb:-1]):
          # Aplicamos fórmula
          p *= self.modelo["condicionales"][clase][i].get(valor, 1e-6)
        
        #Metemos el valor calculado de la probabildiad de la clase en el diccionario
        probabilidades_clase[clase] = p
      #Obtenemos la clase con mayor probabilidad
      predicciones_clases.append(max (probabilidades_clase, key=probabilidades_clase.get))
    return predicciones_clases

        
            
    

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

        predicciones = np.ndarray(datos.datos.shape[0])

        # Clasificamos cada dato
        for sampleTest in datos:
          for i, sampleTrain in enumerate(self.datosTrain.datos):
            distancia = 0

            np.sum(sampleTest - sampleTrain)
