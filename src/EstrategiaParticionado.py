"""
Autores: Claudia Cuevas Ruano, Pablo Tejero Lascorz
Pareja: 03
"""
from abc import ABCMeta,abstractmethod
import random
from Datos import Datos

class Particion():

  # Esta clase mantiene la lista de indices de Train y Test para cada particion del conjunto de particiones  
  def __init__(self, indicesTrain: list = [], indicesTest: list = []):
    self.indicesTrain=indicesTrain
    self.indicesTest=indicesTest

#####################################################################################################

class EstrategiaParticionado:

  # Clase abstracta
  __metaclass__ = ABCMeta

  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor 

  @abstractmethod
  def creaParticiones(self, datos: Datos, seed=None):
    pass
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):
  def __init__(self, numeroEjecuciones: int, proporcionTest: float):
    self.numeroEjecuciones = numeroEjecuciones
    self.proporcionTest = proporcionTest
    self.particiones = []

  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el n�mero de ejecuciones deseado
  # Devuelve una lista de particiones (clase Particion)
  def creaParticiones(self, datos: Datos, seed=None):
    nrows = datos.datos.shape[0]
    indices = list(range(nrows))

    random.seed(seed)
    random.shuffle(indices)

    limit = int(nrows * (1 - self.proporcionTest))

    self.particiones.append(Particion(
      indices[ : limit],
      indices[limit : ]
    ))


#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):
  def __init__(self, numeroParticiones: int):
    self.numeroParticiones = numeroParticiones
    self.particiones = []

  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  def creaParticiones(self, datos: Datos, seed=None):
    nrows: int = datos.datos.shape[0]

    # Crear una permutacion de las filas del dataset
    indices = list(range(nrows))

    random.seed(seed)
    random.shuffle(indices)

    # Filas por cada fold
    rows_per_fold         = nrows // self.numeroParticiones

    # Si la divison entre el numero de filas del dataset y el numero de
    # particiones no es entera, entonces debemos distribuir el resto de
    # manera equitativa entre todas las particiones
    folds_with_extra_rows = nrows % self.numeroParticiones

    start_row = 0
    stop_row = 0
    for i in range(self.numeroParticiones):
      start_row = stop_row
      stop_row = start_row + rows_per_fold

      # Si el resto es mayor que 0, es igual al numero de particiones que
      # deberan acoger una fila extra.
      if i < folds_with_extra_rows:
        stop_row += 1

      self.particiones.append(
        Particion(
          indicesTrain=indices[0 : start_row] + indices[stop_row : ],
          indicesTest=indices[start_row : stop_row]
        )
      )
