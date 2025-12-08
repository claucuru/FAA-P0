"""
Autores: Claudia Cuevas Ruano, Pablo Tejero Lascorz
Pareja: 03
"""
from typing import Optional
import numpy as np
from sklearn.cluster import KMeans
from clustering import Agrupador


class SklearnKMeans(Agrupador):
    def __init__(self, k: int, max_iter: int, limite: Optional[float] = 0.0001, seed: Optional[int] = None):
        self.k = k
        self.max_iter = max_iter
        self.limite = limite
        self.seed = seed
        self.sse = np.inf
        self.labels = None

        self._modelo = KMeans(n_clusters=k, max_iter=max_iter, tol=limite, random_state=seed)

    def agrupar(self, datos: np.ndarray):
        self._modelo.fit(datos)
        self.sse = self._modelo.inertia_
        self.labels = self._modelo.labels_
