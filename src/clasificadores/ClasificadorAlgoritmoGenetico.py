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
        self.best_fitness = None
        self._fitness = -np.inf

        self.encoder = OneHotEncoder()


    def entrenamiento(self, datos: Datos):
        self.encoder.fit(datos.datos) 
        datosTrain = self.encoder.transform(datos.datos)
        len_regla = len(datosTrain[0])

        # Poblacion -> Conjunto de individuos.
        # Individuo -> Un cromosoma.
        # Cromosoma -> Conjunto de reglas o genes.
        # Regla     -> Equivalente a una fila del dataset.
        # Gen       -> Una regla.
        # Alelo     -> Valor posible para un gen. Hay 2^len_regla posibles alelos. 

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
            n_reglas = random.randint(1, self.max_reglas)
            for _ in range(n_reglas):
                # Creamos un gen o regla como una combinacion aleatoria de valores para
                # cada atributo
                regla = []

                for attr in datos.datos.columns.values:
                    # Escogemos un valor aleatorio para el atributo
                    value = random.choice(datos.diccionario[attr])
                    regla.append(value)

                individuo.append(individuo)

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

            # Lista de descendientes
            descendientes = []

            # Arcane s2ep9
            # TODO: Escoger una regla del progenitor para el cruce
            for i in range(progenitores_seleccionados, step=2):
                p1 = progenitores_seleccionados[i]
                p2 = progenitores_seleccionados[i + 1]

                punto_cruce = random.randint(1, len_regla - 1)

                s1 = p1[:punto_cruce] + p2[punto_cruce:]
                s2 = p1[punto_cruce:] + p2[:punto_cruce]

                descendientes.append(s1)
                descendientes.append(s2)

            # Mutacion: cada bit tiene una probabilidad de mutar
            # TODO: Comprobar
            for individuo in poblacion:
                for regla in individuo:
                    for bit in regla:
                        prob_mutacion = random.randint(len_regla * len(poblacion))

                        if prob_mutacion == 1:
                            # mutacion
                            bit = (~bit) & 1

                    descendientes.append(bit)

            # Seleccion de supervivientes
            poblacion.clear()

            n_descendientes = len(descendientes)
            n_mejores = n_descendientes * self.elitismo

            fitness = [self._fitness(individuo) for individuo in descendientes]
            idx_n_mejores = np.argsort(fitness)[n_descendientes - n_mejores:]

            for i in idx_n_mejores:
                poblacion.append(descendientes.pop(i))

            poblacion += np.random.shuffle(descendientes)[self.n_individuos - n_mejores]

            for ind in poblacion:
                f = self._fitness(ind)
                if f > self.best_fitness:
                    self.best_fitness = f
                    self.best_individual = ind

            n_epochs += 1


    def clasifica(self, datos: Datos):
        datosTest = self.encoder.transform(datos.datos)
        return self._fitness(self.best_individual, datosTest)
         


    def _fitness(self, individuo, datos: Datos):
        datosTrain = self.encoder.transform(datos.datos)

        # cuantas instancias del dataset cumplen al menos una regla del individuo
        aciertos = 0
        # Para cada instancia del dataset
        for i in range(datos.shape[0]):
            x = datos[i]
            coincide = False
            for regla in individuo:
                if np.array_equal(x, regla):
                    coincide = True
                    break
            if coincide:
                aciertos += 1
        return aciertos / datos.shape[0]
            

