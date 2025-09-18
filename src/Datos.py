# -*- coding: utf-8 -*-

# coding: utf-8
import pandas as pd
import numpy as np

class Datos:
    # Constructor: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero, delimiter: str = ","):
        self.datos: pd.DataFrame = pd.read_csv(nombreFichero, delimiter=delimiter)

        #Numero filas y columnas
        print(self.datos.shape)

        nominalAttrs = set(self.datos.select_dtypes(include='object', exclude=None).columns.values)
        self.nominalAtributos: list[bool] = [ True if attr in nominalAttrs else False for attr in self.datos.columns ]


    # Devuelve el subconjunto de los datos cuyos �ndices se pasan como argumento
    def extraeDatos(self,idx):
        pass


if __name__ == "__main__":
    dataset = Datos("./datasets/heart.csv")

    print(dataset.nominalAtributos)
