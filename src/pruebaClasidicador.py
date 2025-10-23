from abc import ABCMeta,abstractmethod
from collections import Counter
from enum import Enum
import numpy as np
from Datos import Datos
from EstrategiaParticionado import EstrategiaParticionado


def entrenamiento(datosAll):
    # datos.ndim
    #datos.shape[0] numero de filas del data set
    #datos.shape[1] número de atributos (+ clase)


    #Saco una lista con todos los valores de clase:
    lista_clases = datosAll.datos.iloc[:, -1]

    #Cuento los valores de cada clase
    n_c = Counter(lista_clases)
    num_tot_clase = {int(k): v for k, v in n_c.items()}

    #Sacar el valor de numero de veces que aparece la clase entre el numero total de filas (shape[0]) => PRIORI

    prioris = {}
    for num in num_tot_clase:
        prioris[num] = num/datosAll.datos.shape[0]

    #Saco las clases que hay
    clases = np.unique(lista_clases)

    # Diccionario con probabilidades condicionadas
    condicionales = {c: {} for c in clases}

    datos = datosAll.datos.shape[0]
    n_attrs = datosAll.datos.shape[1]

    for clase in clases:
        #Sacamos las filas de atributos pertenecientes a esa clase:
        filas_clase = datosAll.datos[datosAll.datos.iloc[:, -1] == clase]
        n_filas = len(filas_clase)

        for i in range (n_attrs): # ALTURA
            #Distintos valores que puede tener un atributo i en la clase c y las veces que aparece ese valor en esa clase
            valores, counts = np.unique(filas_clase.iloc[:, i], return_counts=True)
            
            # Obtenemos el número de valores posibles del atributo i
            k = len(np.unique(datosAll.datos.iloc[:, i])) # Número de valores (3) -> (1.5, 1.7, 1.8) 
            #Diccionario para atributo i en la clase c
            condicionales[clase][i] = {}


            for v in np.unique(datosAll.datos[:,i]):
                # Recorremos todos los posibles valores del atributo y almacenamos el número de veces que aparece en la clase
                if v in valores:
                    count_v = counts[valores == v][0]
                else:
                    count_v = 0

                condicionales[clase][i][v] = (count_v + 1) / (n_c + k)

    modelo = {"prioris": prioris, "condicionales":condicionales}
    return modelo


# datosTest: matriz numpy o dataframe con los datos de validaci�n
# nominalAtributos: array bool con la indicatriz de los atributos nominales
# diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables
# devuelve un numpy array o vector con las predicciones (clase estimada para cada fila de test)
def clasifica(datosAll, modelo):
    predicciones_clases = []

    for atrb in datosAll.datos:
        probabilidades_clase = {}   
        # Recorremos todas las clases
        for clase in modelo["prioris"]:
            #Cogemos la prioridad a priori de la clase
            p = modelo["prioris"][clase]

            # Para todos los atributos menos la clase
            for i, valor in enumerate(datosAll.datos.datos.iloc[atrb:-1]):
                # Aplicamos fórmula
                p *= modelo["condicionales"][clase][i].get(valor, 1e-6)

            #Metemos el valor calculado de la probabildiad de la clase en el diccionario
            probabilidades_clase[clase] = p
        #Obtenemos la clase con mayor probabilidad
        predicciones_clases.append(max (probabilidades_clase, key=probabilidades_clase.get))
    return predicciones_clases


if __name__ == '__main__':
    dataset_test=Datos('../datasets/heart-test.csv')
    print(dataset_test)
    dataset_train=Datos('../datasets/heart-train.csv')
    # estrategia=EstrategiaParticionado.ValidacionSimple(5, 0.8)

    modelo = entrenamiento(dataset_test)
    print(modelo)
    predicciones = clasifica(dataset_test, modelo)
    print(predicciones)





