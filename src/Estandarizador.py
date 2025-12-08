"""
Autores: Claudia Cuevas Ruano, Pablo Tejero Lascorz
Pareja: 03
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Datos import Datos


class Estandarizador():
    def __init__(self, datosReferencia: Datos, media: bool = True, std: bool = True):
        # Crear y ajustar estandarizador y codificador
        self.scaler = StandardScaler(with_mean=media, with_std=std)

        # Se estandariza todo menos la clase
        self.scaler.fit(datosReferencia.datos.iloc[:,:-1])


    def estandarizarDatos(self, datos: pd.DataFrame):
        return pd.DataFrame(
            self.scaler.transform(datos.iloc[:,:-1]),
            dtype=np.float64
        )