from abc import ABCMeta,abstractmethod
import random
from sklearn.model_selection import KFold

class Particion():

  # Esta clase mantiene la lista de �ndices de Train y Test para cada partici�n del conjunto de particiones  
  def __init__(self, indicesTrain: list = [], indicesTest: list = []):
    self.indicesTrain=indicesTrain
    self.indicesTest=indicesTest

#####################################################################################################

class EstrategiaParticionado:

  # Clase abstracta
  __metaclass__ = ABCMeta

  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor 

  @abstractmethod
  def creaParticiones(self,datos,seed=None):
    pass
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):
  def __init__(self, numeroEjecuciones: int, proporcionTest: int):
    self.numeroEjecuciones = numeroEjecuciones
    self.proporcionTest = proporcionTest
    self.particiones = []

  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el n�mero de ejecuciones deseado
  # Devuelve una lista de particiones (clase Particion)
  def creaParticiones(self,datos,seed=None):
    nrows = datos.shape[0]
    indices = list(range(nrows))

    random.seed(seed)
    random.shuffle(indices)

    self.particiones.append(Particion(
      indices[ : nrows * self.proporcionTest],
      indices[nrows * self.proporcionTest : ]
    ))


#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):
  def __init__(self, numeroParticiones: int):
    self.numeroParticiones = numeroParticiones
    self.particiones = []

  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  def creaParticiones(self,datos,seed=None):
    kf = KFold(n_splits=self.numeroParticiones, shuffle=True, random_state=seed)

    for train_index, test_index in kf.split(datos):
      self.particiones.append(
        indicesTrain=train_index,
        indicesTest=test_index
      )
