import random
from sklearn.preprocessing import OneHotEncoder
from clasificadores.Clasificador import Clasificador
from Datos import Datos


class ClasificadorAG(Clasificador):
    def __init__(self, elitismo: float, n_individuos: int, max_epochs: int, max_reglas: int):
        self.elitismo = elitismo
        self.n_individuos = n_individuos
        self.max_epochs = max_epochs
        self.max_reglas = max_reglas

        self.encoder = OneHotEncoder()


    def entrenamiento(self, datos: Datos):
        self.encoder.fit(datos.datos)
        datosTest = self.encoder.transform(datos.datos)


        # Escogemos una poblacion inicial. La poblacion es un conjunto de individuos cuyo
        # tamano viene dado por self.n_individuos. Cada individuo es un conjunto de uno o
        # varias reglas o cromosomas, cuyo numero viene dado por self.max_reglas. Una regla
        # es una posible solucion valida para el problema. Cada fila del dataset es una 
        # regla. Las reglas iniciales, sin embargo, no se escogen del dataset sino que se
        # crean de manera aleatoria, tomando para cada atributo un valor aleatorio de los
        # posibles valores.
        poblacion = []

        for individuo in range(self.n_individuos):
            individuo = []

            for cromosoma in range(self.max_reglas):
                # Creamos un cromosoma como una combinación aleatoria de valores
                # para cada atributo
                cromosoma = []

                for attr in datos.datos.columns.values:
                    # Escogemos un valor aleatorio
                    value = random.choice(datos.diccionario[attr])
                    cromosoma.append(value)

                individuo.append(cromosoma)

            poblacion.append(individuo)


        n_epochs = 0
        while n_epochs < self.max_epochs:
            # Seleccion de progenitores para la descendencia. Go spin the wheel.
            # Recomendado: https://es.piliapp.com/random/wheel/
            # Aplicamos la funcion F a cada individuo de la muestra. Dividimos el
            # resultado entre la suma de las funciones F de cada individuo.
            results_f = [self._fitness(individuo) for individuo in poblacion]
            sum_fitness = sum(results_f)
            results_f = [fitness / sum_fitness for fitness in results_f]

            n_epochs += 1


    def clasifica(self, datos: Datos):
        datosTrain = self.encoder.transform(datos.datos)


    def _fitness(self, individuo):
        pass