"""
Autores: Claudia Cuevas Ruano, Pablo Tejero Lascorz
Pareja: 03
"""
from .Clasificador import Clasificador
from collections import Counter
import numpy as np
from Datos import Datos
from math import sqrt, pi, exp


SQRT_2_PI = sqrt(2 * pi)

class ClasificadorNaiveBayes(Clasificador):

  # TODO: esta funcion debe ser implementada en cada clasificador concreto. Crea el modelo a partir de los datos de entrenamiento
  # datosTrain: matriz numpy o dataframe con los datos de entrenamiento
  # nominalAtributos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables
  def entrenamiento(self, datos: Datos):
    n_muestras = datos.datos.shape[0]
    n_attrs = datos.datos.shape[1] - 1


    # Lista de las clases que hay
    clases_muestras = datos.datos.iloc[:, -1]
    clases = np.unique(clases_muestras)

    # Cuento el numero de muestras para cada clase y calculo las probabilidades
    # a priori de cada clase como el numero de veces que aparece una muestra con
    # esa clase entre el numero de muestras totales.
    prioris = {
        int(clase): (n / n_muestras) for clase, n in Counter(clases_muestras).items()
    }

    # Diccionario con probabilidades condicionadas
    condicionales = {c: {} for c in clases}

    for clase in clases:
        #Sacamos las filas de atributos pertenecientes a esa clase:
        filas_clase = datos.datos[datos.datos.iloc[:, -1] == clase]
        n_filas = len(filas_clase)

        for i in range(n_attrs):
            if datos.nominalAtributos[i]:
                #Distintos valores que puede tener un atributo i en la clase c y las veces que aparece ese valor en esa clase
                valores, counts = np.unique(filas_clase.iloc[:, i], return_counts=True)

                # Obtenemos el número de valores posibles del atributo i
                k = len(np.unique(datos.datos.iloc[:, i])) # Número de valores (3) -> (1.5, 1.7, 1.8) 
                #Diccionario para atributo i en la clase c
                condicionales[clase][i] = {}


                for v in np.unique(datos.datos.iloc[:,i]):
                    # Recorremos todos los posibles valores del atributo y almacenamos el número de veces que aparece en la clase
                    if v in valores:
                        count_v = counts[valores == v][0]
                    else:
                        count_v = 0
                    # Corrección de Laplace
                    condicionales[clase][i][v] = (count_v + 1) / (n_filas + k)
                    
            else:
                #Selecciona la columna i de la clase
                mu = np.mean(filas_clase.iloc[:,i])
                sigma = np.std(filas_clase.iloc[:, i], ddof=1) #Muestra
                condicionales[clase][i] = {"mean": mu, "std": sigma}

    self._prioris = prioris
    self._condicionales = condicionales


  # datosTest: matriz numpy o dataframe con los datos de validaci�n
  # nominalAtributos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables
  # devuelve un numpy array o vector con las predicciones (clase estimada para cada fila de test)
  def clasifica(self, datos: Datos):
    predicciones_clases = []

    for _, fila in datos.datos.iterrows():
        probabilidades_clase = {}

        # Recorremos todas las clases
        for clase, priori in self._prioris.items():

            # Para todos los atributos menos la clase
            for i, valor in enumerate(fila[:-1]):
                #Si son campos nominales entonces no habrá que aplicar la expresión de la distribución normal
                if datos.nominalAtributos[i]:
                    # Aplicamos fórmula
                    posteriori = priori * self._condicionales[clase][i].get(valor, 1e-6)
                else:
                    mu = self._condicionales[clase][i]["mean"]
                    sigma = self._condicionales[clase][i]["std"]
                    if sigma == 0:
                        sigma = 1e-6
                    posteriori = (1.0 / (SQRT_2_PI * sigma)) * exp(-((valor - mu) ** 2) / (2 * sigma ** 2))

            #Metemos el valor calculado de la probabildiad de la clase en el diccionario
            probabilidades_clase[clase] = posteriori
        #Obtenemos la clase con mayor probabilidad
        predicciones_clases.append(max (probabilidades_clase, key=probabilidades_clase.get))
    return predicciones_clases


  def limpiar_numpy(self, array):
      if isinstance(array, dict):
          return {self.limpiar_numpy(k): self.limpiar_numpy(v) for k, v in array.items()}
      elif isinstance(array, (list, tuple, set)):
          return type(array)(self.limpiar_numpy(x) for x in array) 
      elif isinstance(array, (np.int32, np.int64, np.integer)):
          return int(array)
      elif isinstance(array, (np.float32, np.float64, np.floating)):
          return float(array)
      elif isinstance(array, np.ndarray):
          return array.tolist()
      else:
          return array