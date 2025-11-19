import numpy as np
import pandas as pd

from Datos import Datos

# 1. Elegir el numero de clusters k:
    #Para ello Método del codo. Consiste en representar gráficamente el modelo (varianzas num cluster) y ver el momento en el que ya no se mejora considerablemente el modelo
# 2. Seleccionar aleatoriamente los centroides: dónde está el centro de cada clúster
# 3. Se asigna cada punto al centroide más cercano usando la distancia euclidea 
# 4. Se evalua la calidad de los clústers (se usa función objetivo)
# 5. Se recalculan los centroides (punto medio entre los datos de cada grupo)
# 6. Se repiten 3 y 4 hasta que el algoritmo converga según los pasos establecidos por nosotros:
    # Híper parámetros: 
    # 6.1 Número de clústers (en cuántos grupos vamos a separar los datos)
    # 6.2 Número de iteraciones máximas (como criterio de convergencia del algoritmo)
    # 6.3 Límite de la métrica de evaluación (otro criterio de convergencia)

# LOS RESULTADOS PUEDEN VARIAR!

class KMeans:
    def __init__(self, k: int, max_iter: int, limite: float, seed: int = None):
        self.k = k
        self.max_iter = max_iter
        self.limite = limite
        self.seed = seed
        self.centroides = None
        self.labels = None # que cluster pertenece a cada patrón
        self.SSE= None
        self.k = k

    # Calcula la distancia euclidea entre un punto y todos los centroides
    def distancia_euclidea (self, punto, centroides):
        # return np.sqrt(np.sum(punto - centroides) ** 2, axis=1)
        return np.sqrt(np.sum((punto[:, np.newaxis, :] - centroides[np.newaxis, :, :]) ** 2, axis=2))
        
    
    # Selecciona aleatoriamente los centroides iniciales
    def inicializar_centroides (self, datos):
        if self.seed is not None:
            np.random.seed(self.seed)
        
        num_filas = datos.shape[0]
        # indices aleatorios con replace a false para que no haya centroides duplicados
        indices = np.random.choice(num_filas, self.k, replace=False)
        return datos[indices, :]
    
    def recalcular_centroides (self, datos, labels):
        nuevos_centroides = np.zeros((self.k, datos.shape[1]))
        # Por cada clúster
        for cluster_index in range(self.k):
            puntos = datos[labels == cluster_index] # todos los puntos del cluster i
            # Si hay puntos en ese cluster se saca la media de las columnas 
            if len(puntos) > 0:
                nuevos_centroides[cluster_index] = puntos.mean(axis=0)
            else:
            # Si no hay puntos en ese cluster se coloca el centroide en un punto aleatorio
                nuevos_centroides[cluster_index] = datos[np.random.choice(len(datos))]
            
        return nuevos_centroides
    
    def calcular_SSE (self, dataset, labels, centroides):
        sse = 0.0
        #Por cada cluster
        for cluster_index in range (self.k):
            puntos_cluster = dataset[labels == cluster_index] # sacamos todos los puntos del cluster i
            if len(puntos_cluster) > 0: #si hay puntos en ese cluster se calcula el error
                sse += np.sum((puntos_cluster - centroides[cluster_index])** 2) 
                 
        return sse

    def KMeansAlgorithm (self, datos):
        datos = np.asarray(datos, dtype=float)

        self.centroides = self.inicializar_centroides(datos)

        # Solo hacemos como máximo las iteraciones indicadas
        for _ in range (self.max_iter):
            #Calculamos distancias y elegimos el centroide más cercano
            distancias = self.distancia_euclidea(datos, self.centroides)
            # indice del centroide más cercano
            labels = np.argmin(distancias, axis=1)

            # Calculamos los neuvs centroides
            nuevos_centroides = self.recalcular_centroides(datos, labels)

            sse_actual = self.calcular_SSE(datos, labels, nuevos_centroides)

            # Si ya existia un SSE anterior, comprobamos la mejora
            if self.SSE is not None:
                mejora = abs(self.SSE - sse_actual)
                # Si la mejora es menor que el limite que le hemos puesto, hemos convergido
                if mejora < self.limite:
                    self.centroides = nuevos_centroides
                    self.labels = labels
                    self.SSE = sse_actual
                    break
            # Si no es la primera iteracion o el límite no se ha cumplido, seguimos
            self.SSE = sse_actual
            self.centroides = nuevos_centroides
            self.labels = labels
        
        return self
            

        

        
    
