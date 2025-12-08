"""
Autores: Claudia Cuevas Ruano, Pablo Tejero Lascorz
Pareja: 03
"""
from typing import Optional
import numpy as np
from clustering import Agrupador


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

class KMeans(Agrupador):
    def __init__(self, k: int, max_iter: int, limite: float, seed: Optional[int] = None):
        self.k = k
        self.max_iter = max_iter
        self.limite = limite
        self.seed = seed
        self.centroides = None
        self.labels = None # que cluster pertenece a cada patrón
        self.SSE= np.inf
        self.k = k


    # Selecciona aleatoriamente los centroides iniciales
    def _inicializar_centroides(self, datos: np.ndarray) -> np.ndarray:
        """
        Escoge unos puntos aleatorios que se convertiran en los primeros
        centroides
        """
        np.random.seed(self.seed)
        
        num_filas = datos.shape[0]

        # indices aleatorios con replace a false para que no haya centroides duplicados
        indices = np.random.choice(num_filas, self.k, replace=False)

        return datos[indices, :]


    def _recalcular_centroides(self, datos: np.ndarray, labels):
        """
        Recalcula la posicion de los centroides.
        """
        nuevos_centroides = np.zeros((self.k, datos.shape[1]))

        # Por cada clúster
        for cluster_index in range(self.k):
            # Obtener todos los puntos del cluster
            puntos = datos[labels == cluster_index]

            # Si hay puntos en ese cluster se saca la media de las columnas. si no hay
            # puntos en el cluster, se recoloca el centroide en un punto aleatorio.
            if len(puntos) > 0:
                nuevos_centroides[cluster_index] = puntos.mean(axis=0)
            else:
                nuevos_centroides[cluster_index] = datos[np.random.choice(len(datos))]
            
        return nuevos_centroides


    def _calcular_SSE(self, datos: np.ndarray, labels, centroides: np.ndarray):
        """
        Calcula la suma de errores al cuadrado
        """
        sse = 0.0
        #Por cada cluster
        for cluster_index in range(self.k):
            puntos_cluster = datos[labels == cluster_index] # sacamos todos los puntos del cluster i
            if len(puntos_cluster) > 0: #si hay puntos en ese cluster se calcula el error
                sse += np.sum((puntos_cluster - centroides[cluster_index])** 2)

        return sse

    def agrupar(self, datos: np.ndarray):
        # Escogemos de forma aleatoria unos centroides iniciales
        self.centroides = self._inicializar_centroides(datos)

        # Solo hacemos como máximo las iteraciones indicadas
        for _ in range (self.max_iter):
            #Calculamos distancias y elegimos el centroide más cercano
            distancias = np.sqrt(np.sum((datos[:, np.newaxis, :] - self.centroides[np.newaxis, :, :]) ** 2, axis=2))

            # indice del centroide más cercano
            labels = np.argmin(distancias, axis=1)

            # Calculamos los neuvs centroides
            nuevos_centroides = self._recalcular_centroides(datos, labels)

            sse_actual = self._calcular_SSE(datos, labels, nuevos_centroides)

            # Comprobamos el margen de disminucion del error
            mejora = np.abs(self.SSE - sse_actual)

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
