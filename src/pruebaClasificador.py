from abc import ABCMeta,abstractmethod
from math import sqrt, pi, exp
from collections import Counter
from enum import Enum
import numpy as np
from Datos import Datos
from EstrategiaParticionado import EstrategiaParticionado
import json


def entrenamiento(datosAll):
    # datos.ndim
    #datos.shape[0] numero de filas del data set
    #datos.shape[1] número de atributos (+ clase)


    #Saco una lista con todos los valores de clase:
    lista_clases = datosAll.datos.iloc[:, -1]
    #Saco las clases que hay
    clases = np.unique(lista_clases)

    #Cuento los valores de cada clase
    n_c = Counter(lista_clases)
    num_tot_clase = {int(k): v for k, v in n_c.items()}

    #Sacar el valor de numero de veces que aparece la clase entre el numero total de filas (shape[0]) => PRIORI
    prioris = {}
    for num in num_tot_clase:
        prioris[num] = num_tot_clase[num]/datosAll.datos.shape[0]

    # Diccionario con probabilidades condicionadas
    condicionales = {c: {} for c in clases}

    datos = datosAll.datos.shape[0]
    n_attrs = datosAll.datos.shape[1] - 1

    for clase in clases:
        #Sacamos las filas de atributos pertenecientes a esa clase:
        filas_clase = datosAll.datos[datosAll.datos.iloc[:, -1] == clase]
        n_filas = len(filas_clase)

        for i in range (n_attrs): # ALTURA
            if datosAll.nominalAtributos[i]:            
                #Distintos valores que puede tener un atributo i en la clase c y las veces que aparece ese valor en esa clase
                valores, counts = np.unique(filas_clase.iloc[:, i], return_counts=True)

                # Obtenemos el número de valores posibles del atributo i
                k = len(np.unique(datosAll.datos.iloc[:, i])) # Número de valores (3) -> (1.5, 1.7, 1.8) 
                #Diccionario para atributo i en la clase c
                condicionales[clase][i] = {}


                for v in np.unique(datosAll.datos.iloc[:,i]):
                    # Recorremos todos los posibles valores del atributo y almacenamos el número de veces que aparece en la clase
                    if v in valores:
                        count_v = counts[valores == v][0]
                    else:
                        count_v = 0

                    condicionales[clase][i][v] = (count_v + 1) / (n_filas + k)
            else:
                #Selecciona la columna i de la clase
                mu = np.mean(filas_clase.iloc[:,i])
                sigma = np.std(filas_clase.iloc[:, i], ddof=1) #Muestra
                condicionales[clase][i] = {"mean": mu, "std": sigma}

    modelo = {"prioris": prioris, "condicionales":condicionales}
    return modelo

def gaussian(x, mu, sigma):
    if sigma == 0:
        sigma = 1e-6
    return (1.0 /(sqrt(2 * pi) * sigma)) * exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# datosTest: matriz numpy o dataframe con los datos de validaci�n
# nominalAtributos: array bool con la indicatriz de los atributos nominales
# diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables
# devuelve un numpy array o vector con las predicciones (clase estimada para cada fila de test)
def clasifica(datosAll, modelo):
    predicciones_clases = []

    for _, fila in datosAll.datos.iterrows():
        probabilidades_clase = {}   
        # Recorremos todas las clases
        for clase in modelo["prioris"]:
            #Cogemos la prioridad a priori de la clase
            p = modelo["prioris"][clase]

            # Para todos los atributos menos la clase
            for i, valor in enumerate(fila[:-1]):
                #Si son campos nominales entonces no habrá que aplicar la expresión de la distribución normal
                if datosAll.nominalAtributos[i]:
                    # Aplicamos fórmula
                    p *= modelo["condicionales"][clase][i].get(valor, 1e-6)
                else:
                    mu = modelo["condicionales"][clase][i]["mean"]
                    sigma = modelo["condicionales"][clase][i]["std"]
                p *= gaussian(valor, mu, sigma)

            #Metemos el valor calculado de la probabildiad de la clase en el diccionario
            probabilidades_clase[clase] = p
        #Obtenemos la clase con mayor probabilidad
        predicciones_clases.append(max (probabilidades_clase, key=probabilidades_clase.get))
    return predicciones_clases

def limpiar_numpy(array):
    if isinstance(array, dict):
        return {limpiar_numpy(k): limpiar_numpy(v) for k, v in array.items()}
    elif isinstance(array, (list, tuple, set)):
        return type(array)(limpiar_numpy(x) for x in array) 
    elif isinstance(array, (np.int32, np.int64, np.integer)):
        return int(array)
    elif isinstance(array, (np.float32, np.float64, np.floating)):
        return float(array)
    elif isinstance(array, np.ndarray):
        return array.tolist()
    else:
        return array
    


if __name__ == '__main__':
    dataset_test=Datos('../datasets/heart-test.csv')
    
    dataset_train=Datos('../datasets/heart-train.csv')
    # estrategia=EstrategiaParticionado.ValidacionSimple(5, 0.8)

    print(dataset_test.nominalAtributos)
    print("Diccionario:")
    print(dataset_test.diccionario)
    print("\nDatos:")
    print(dataset_test.datos)
    print("Extraer datos: \n", dataset_test.extraeDatos(1))


    modelo = entrenamiento(dataset_train)
    modelo_limpio = limpiar_numpy(modelo)
    condicionales_limpios = limpiar_numpy(modelo["condicionales"])

    for clase, attrs in modelo_limpio["condicionales"].items():
        print(f"\nClase {clase:}")
        for i, valores in attrs.items():
            print(f"    Atributo {i}: {valores}")

    

    # print(json.dumps(condicionales_limpios[0][1], indent=2))
    # print("Prioris:")
    # print(modelo_limpio["prioris"])
    # print("\nCondicionales:")
    # print(modelo_limpio["condicionales"])
    predicciones = clasifica(dataset_test, modelo_limpio)
    print("\nPredicciones:")
    print(predicciones)






