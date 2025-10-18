
from collections import Counter
import numpy as np
from abc import ABCMeta,abstractmethod
from enum import Enum

datos = np.array([[1, 1, 2, 2], [3, 3, 4, 4], [2, 2, 2, 2]])

# Obtengo las clases cogiendo el último valor de cada fila
lista_clases = datos[:, -1]


# Cuento los valores de cada clase
n_c= Counter(lista_clases)

# Paso los valores a enteros
num_tot_clase = {int(k): v for k, v in n_c.items()}

print(num_tot_clase)

prioris = {}
for clase, num in num_tot_clase.items():
    prioris[clase] = num / len(num_tot_clase)
    
print(prioris)
print(len(num_tot_clase))