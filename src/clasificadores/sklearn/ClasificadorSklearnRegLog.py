"""
Autores: Claudia Cuevas Ruano, Pablo Tejero Lascorz
Pareja: 03
"""
from sklearn.linear_model import LogisticRegression
from clasificadores import Clasificador
from Datos import Datos


class ClasificadorSklearnRegLog(Clasificador):
    def __init__(self, eta: int, epocas: int, seed: int = 33):
        self._n_attrs = -1
        self._model = LogisticRegression(random_state=seed, max_iter=epocas)


    def entrenamiento(self, datos: Datos):
        self._n_attrs = datos.datos.shape[1] - 1

        attrs = datos.datos.iloc[:,:self._n_attrs]
        clases = datos.datos.iloc[:,-1]

        self._model.fit(attrs, clases)


    def clasifica(self, datos: Datos):
        return self._model.predict(datos.datos.iloc[:,:self._n_attrs])
