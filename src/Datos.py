"""
Autores: Claudia Cuevas Ruano, Pablo Tejero Lascorz
Pareja: 03
"""
# -*- coding: utf-8 -*-
import pandas as pd


class Datos:
    # Constructor: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero, delimiter: str = ","):
        """
        Lee un fichero de datos y convierte los atributos nominales en numericos.
        Asigna las siguientes variables:
           - self.datos (`pandas.Dataframe`): Dataset
           - self.nominalAtributos (`list[bool]`): Para cada atributo contiene `True` si
                era originalmente nominal y `False` si es numerico.
           - self.diccionario (`dict`): Registra la conversion de atributos nominales en
                numericos. Las claves del diccionario son los nombres de los atributos
                numericos (ejemplo: "Animal"), y los valores son otro diccionario que
                relaciona los posibles valores del atributo (ejemplo: "Gato" y "Perro")
                con numeros enteros positivos empezando por 0 que se asignan por orden
                alfabetico (0 y 1 para el ejemplo anterior, respectivamente).
        """
        self.datos: pd.DataFrame = pd.read_csv(nombreFichero, delimiter=delimiter)

        # Nombres de las columnas con atributos nominales
        nominalAttrs = set(self.datos.select_dtypes(include='object', exclude=None).columns.values)

        # Lista que, para cada atributo, contiene True si es nominal y False si no lo es
        self.nominalAtributos: list[bool] = [ True if attr in nominalAttrs else False for attr in self.datos.columns ]

        # Registrar la conversion de atributos nominales en numericos
        self.diccionario = dict()
        for attr in nominalAttrs:
            self.diccionario[attr] = dict()
            ordenados = sorted(set(self.datos[attr]))

            for i, elem in enumerate(ordenados):
                self.diccionario[attr][elem] = i

        # Sustituimos en el dataset los valores nominales por los numerales
        for attrName, column in self.datos.items():
            if attrName not in nominalAttrs:
                continue

            for i, value in enumerate(column):
                column.at[i] = self.diccionario[attrName][value]


    def extraeDatos(self, rowIndex: int):
        """
        Devuelve la fila del dataset cuyo indice se pasa como parametro.
        """
        return self.datos.loc[rowIndex]
