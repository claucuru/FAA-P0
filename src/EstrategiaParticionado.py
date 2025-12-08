"""
Autores: Claudia Cuevas Ruano, Pablo Tejero Lascorz
Pareja: 03
"""
from abc import ABCMeta,abstractmethod
from typing import Optional
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


  def __init__(self):
    self.particiones = []


  @abstractmethod
  def creaParticiones(self, datos: Datos, seed: Optional[int] = None) -> list[Particion]:
    pass
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):
  def __init__(self, numeroEjecuciones: int, proporcionTest: float):
    super().__init__()
    self.numeroEjecuciones = numeroEjecuciones
    self.proporcionTest = proporcionTest


  def creaParticiones(self, datos: Datos, seed=None):
    """
    Crea particiones segun el metodo tradicional de division de los datos
    segun el porcentaje deseado y el numero de ejecuciones deseado.
    
    Devuelve
    --------
      Lista de particiones (clase `Particion`)`con un único elemento.
    """
    # Reset listado de particiones
    self.particiones.clear()

    nrows = datos.datos.shape[0]
    indices = list(range(nrows))

    random.seed(seed)
    random.shuffle(indices)

    limit = int(nrows * (1 - self.proporcionTest))

    self.particiones.append(Particion(
      indices[ : limit],
      indices[limit : ]
    ))

    return self.particiones


#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):
  def __init__(self, numeroParticiones: int):
    super().__init__()
    self.numeroParticiones = numeroParticiones


  def creaParticiones(self, datos: Datos, seed=None):
    """
    Crea particiones segun el metodo de validacion cruzada. El conjunto
    de entrenamiento se crea con los (K-folds - 1) particiones y el de
    test con la particion restante.

    Devuelve
    --------
      Lista de particiones (clase `Particion`).
    """
    # Vaciar listado de particiones
    self.particiones.clear()

    nrows: int = datos.datos.shape[0]
    indices = list(range(nrows))

    random.seed(seed)
    random.shuffle(indices)

    # Filas por cada fold
    rows_per_fold = nrows // self.numeroParticiones

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

    return self.particiones
