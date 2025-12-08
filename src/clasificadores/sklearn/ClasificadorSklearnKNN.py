"""
Autores: Claudia Cuevas Ruano, Pablo Tejero Lascorz
Pareja: 03
"""
from sklearn.neighbors import KNeighborsClassifier
from clasificadores import Clasificador, DistanceMetric
from Datos import Datos


class ClasificadorSklearnKNN(Clasificador):
    def __init__(self, k: int, distanceMetric: DistanceMetric):
        self.k = k
        self.distanceMetric = distanceMetric
        self._n_attrs = -1

        match distanceMetric:
            case DistanceMetric.EUCLIDES:
                sk_metric = "euclidean"
            case DistanceMetric.MANHATTAN:
                sk_metric = "manhattan"
            case _:
                raise ValueError("Unsupported distance metric")

        self._model = KNeighborsClassifier(k, weights="uniform", algorithm="brute", metric=sk_metric)


    def entrenamiento(self, datos: Datos):
        attrs = datos.datos.iloc[:,:-1]
        clases = datos.datos.iloc[:,-1]


        self._n_attrs = attrs.shape[1]
        self._model.fit(attrs, clases)


    def clasifica(self, datos: Datos):
        attrs = datos.datos.iloc[:,:self._n_attrs]
        return self._model.predict(attrs)