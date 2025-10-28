from abc import ABCMeta,abstractmethod
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
  def error(self,datos: Datos, pred: list):
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
    # - Para validacion simple (hearme-out): entrenamos el clasificador con la particion de train
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
    particionado.creaParticiones(dataset, seed)
  
    for particion in particionado.particiones:    
      datosTrain = dataset.particion(particion.indicesTrain)
      datosTest = dataset.particion(particion.indicesTest)

      self.entrenamiento(datosTrain)
      predicciones = self.clasifica(datosTest)
