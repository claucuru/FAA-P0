import random
import numpy as np
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
        datosTrain = self.encoder.transform(datos.datos)
        attrsTrain = datosTrain[:,:-1]
        clasesTrain = datosTrain[:,-1]
        len_regla = len(datosTrain[0])

        # Cromosoma -> Fila dataset. Una regla es un cromosoma.
        # Individuo -> Conjunto de cromosomas.
        # Poblacion -> Conjunto de individuos.

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
                # Creamos un cromosoma como una combinacion aleatoria de valores
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
            
            # Array con los resultados de fitness acumulados
            cum = 0
            cum_f = np.empty(len(results_f))
            for i, result in enumerate(results_f):
                cum_f[i] = result + cum

            progenitores_seleccionados = [
                poblacion[np.searchsorted(cum_f, random.uniform(0, 1)) - 1]
                for _ in range(self.n_individuos)
            ]

            # Arcane s2ep9
            # TODO: Escoger una regla del progenitor para el cruce
            for i in range(progenitores_seleccionados, step=2):
                p1 = progenitores_seleccionados[i]
                p2 = progenitores_seleccionados[i + 1]

                punto_cruce = random.randint(1, len_regla - 1)

                s1 = p1[:punto_cruce] + p2[punto_cruce:]
                s2 = p1[punto_cruce:] + p2[:punto_cruce]

            # Mutacion: cada individuo de la poblacion tiene una probabilidad
            # de mutar
            for individuo in poblacion:
                prob_mutacion = random.randint(len_regla * len(poblacion))

            n_epochs += 1


    def clasifica(self, datos: Datos):
        datosTrain = self.encoder.transform(datos.datos)


    def _fitness(self, individuo):
        pass