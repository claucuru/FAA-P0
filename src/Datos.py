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

        # Nombres de las columnas con atributos nominales
        nominalAttrs = set(self.datos.select_dtypes(include='object', exclude=None).columns.values)

        self.nominalAtributos: list[bool] = [ True if attr in nominalAttrs else False for attr in self.datos.columns ]


        self.diccionario = dict()
        for attr in nominalAttrs:
            self.diccionario[attr] = dict()
            ordenados = sorted(set(self.datos[attr]))

            for i, elem in enumerate(ordenados):
                self.diccionario[attr][elem] = i

        # Iteramos sobre la lista con las cabeceras del dataset que nos indica si la columna es nominal o no. 
        for attrName, column in self.datos.items():
            if attrName not in nominalAttrs:
                continue

            print(attrName)
            for j, value in enumerate(column):
                column.at[j] = self.diccionario[attrName][value]


    # Devuelve el subconjunto de los datos cuyos indices se pasan como argumento
    def extraeDatos(self, rowIndex: int):
        return self.datos.loc[rowIndex]


if __name__ == "__main__":
    dataset = Datos("./datasets/heart-copy.csv")

    print(dataset.nominalAtributos)
    print(dataset.diccionario)

    print(dataset.datos)
    print("Extraer datos: \n", dataset.extraeDatos(2))
