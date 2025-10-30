import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from Datos import Datos


class Estandarizador:
    def __init__(self, datosReferencia: Datos, media: bool = True, std: bool = True, exclude: list[str] = None):
        """
        Crea un estandarizador de datos.

        Parametros
        ----------
        - datosReferencia: [Datos] Dataset con los datos que se usaran como referencia
            para ajustar el estandarizador y procesar los siguientes datos. 
        - media: [bool] Para estandarizar atributos numericos con la media.
        - std: [bool] Para estandarizar atributos numericos con la desviacion
            estandar.
        - exclude: list[str] Lista con los nombres de las columnas de atributos
            nominales que no se desea codificar. Se deberia usar con atributos nominales
            cuantitativos.
        """
        if not exclude:
            exclude = ()

        # Generar listas con los nombres de las columnas de los atributos de
        # cada tipo, necesarias para especificar a scaler y encoder las columnas
        # que deben modificar.
        self.colsAttrsNumericos = []
        self.colsAttrsNominales = []

        # Obtener todos los posibles valores de un atributo nominal para cada
        # atributo nominal
        categorias = []

        for nombreColumna, esNominal in zip(datosReferencia.datos.columns, datosReferencia.nominalAtributos):
            if esNominal and nombreColumna not in exclude:
                # diccionario[nombreColumna] devuelve un diccionario donde las
                # claves son las cadenas de caracteres originales de todos de los
                # posibles valores del atributo
                valores = list(datosReferencia.diccionario[nombreColumna].values())

                # Si solo hay dos valores, se puede considerar el atributo como uno
                # de tipo booleano. Ademas, los clasificadores de sklearn se pueden
                # quejar si pasas una columna de atributos en la que los valores son
                # todos 0.0 o 1.0 porque jaja muy buen diseno!
                if len(valores) == 2:
                    self.colsAttrsNumericos.append(nombreColumna)

                self.colsAttrsNominales.append(nombreColumna)
                categorias.append(valores)

            else:
                self.colsAttrsNumericos.append(nombreColumna)


        # Obtener todos los posibles valores de un atributo nominal para cada
        # atributo nominal
        # categorias = []

        # for nombreColumna in self.colsAttrsNominales:
        #     # diccionario[nombreColumna] devuelve un diccionario donde las
        #     # claves son las cadenas de caracteres originales de todos de los
        #     # posibles valores del atributo
        #     categorias.append(list(datosReferencia.diccionario[nombreColumna].values()))


        # Crear y ajustar estandarizador y codificador
        self.scaler = StandardScaler(with_mean=media, with_std=std)
        self.encoder = OneHotEncoder(sparse_output=False, categories=categorias)

        self.scaler.fit(datosReferencia.datos[self.colsAttrsNumericos])
        self.encoder.fit(datosReferencia.datos[self.colsAttrsNominales])


    def estandarizarDatos(self, datos: pd.DataFrame):
        """
        Estandariza los atributos numericos y aplica una codificacion OneHot a
        los atributos nominales para evitar sesgos y errores de escalas durante
        la clasificacion de los datos.

        Argumentos
        ----------
        - datos: [pandas.DataFrame] Dataset de datos que se quiere estandarizar.

        Devuelve
        --------
        Un nuevo DataFrame con los datos procesados.
        """
        # Estandarizamos solo los atributos numericos. StandardScaler.fit_transform
        # calcula la media y la desviacion estandar de cada atributo y luego aplica
        # la estandarizacion.
        dfNumericos = pd.DataFrame(
            self.scaler.transform(datos[self.colsAttrsNumericos]),
            columns=self.colsAttrsNumericos,
        )

        # Los atributos categoricos o nominales no se deben estandarizar de esta forma.
        # En su lugar, lo adecuado es, para cada atributo, crear una columna por cada
        # posible valor. Por ejemplo, si tenemos un atributo "Animal" que puede tomar
        # los valores "Perro", "Gato" y "Pez", entonces eliminamos la columna "Animal"
        # y creamos tres nuevas columnas en las que el valor de la celda para cada
        # muestra del dataset sera 1 si es ese animal y 0 en caso contrario.
        dfNominales = pd.DataFrame(
            self.encoder.transform(datos[self.colsAttrsNominales]),
            columns=self.encoder.get_feature_names_out(self.colsAttrsNominales),
        )

        return pd.concat((dfNumericos, dfNominales), axis=1)
