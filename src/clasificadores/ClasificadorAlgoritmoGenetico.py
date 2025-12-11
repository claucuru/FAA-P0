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
        self.best_fitness = -np.inf
        self.best_individuo = None

        self.encoder = OneHotEncoder()
        self.num_val_por_atributo = None


    def entrenamiento(self, datos: Datos):
        self.encoder.fit(datos.datos) 
        datosTrain = self.encoder.transform(datos.datos)
        len_regla = len(datosTrain[0])

        self.num_val_por_atributo = [
            len(valores_atributo) for valores_atributo in self.encoder.categories_
        ]

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

                individuo.append(regla)

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
                    self.best_individuo = ind

            n_epochs += 1


    def clasifica(self, datos: Datos):
        datosTest = self.encoder.transform(datos.datos)
        return [ self._predecir_clase_individuo(self.best_individuo, muestra) for muestra in datosTest ]


    def _fitness(self, individuo, datos: np.ndarray):
        # cuantas instancias del dataset cumplen al menos una regla del individuo
        num_aciertos = 0

        # Contabilizar aciertos y errores para cada fila del dataset
        for muestra in datos:
            pred = self._predecir_clase_individuo(individuo, muestra)

            if pred == muestra[-1]:
                aciertos += 1

        return num_aciertos / datos.shape[0]


    def _predecir_clase_individuo(self, individuo: list[np.ndarray], muestra: np.ndarray):
            coincidencias = [0, 0]

            # Analizamos si se cumple alguna regla
            for regla in individuo:
                cumple_regla = True

                col_ini = 0
                col_end = 0
                for num_valores_attr in self.num_val_por_atributo:
                    col_ini += col_end
                    col_end += num_valores_attr

                    cumple_condicion_attr = False

                    for col in range(col_ini, col_end):
                        if muestra[col] == regla[col]:
                            cumple_condicion_attr = True
                            break

                    if not cumple_condicion_attr:
                        cumple_regla = False
                        #break

                if cumple_regla:
                    coincidencias[regla[-1]] += 1


            if coincidencias[0] == coincidencias[1]:
                # Ninguna regla coincide con la muestra
                if coincidencias[0] == 0:
                    return None
                else:
                    pred = random.choice(coincidencias)
            
            return pred


        # Para cada instancia del dataset
        # for i in range(datos.datos.shape[0]):
        #     # Reglas del individuo que coinciden
        #     reglasCoinciden = []
        #     x = datos.datos.iloc[i]
        #     # Verificamos cada regla del individuo
        #     for regla in individuo:
        #         coincide = True
        #         # Compara cada atributo
        #         for j, attr in enumerate (atributos[:-1]):
        #             if x[attr] != regla[attr]:
        #                 coincide = False
        #                 break
        #         if coincide:
        #             # metemos la prediccion, que es el último elemento
        #             reglasCoinciden.append(regla[-1])
            
        #     # Si no coincide ninguna, pasamos a la siguiente instancia
        #     if not reglasCoinciden:
        #         continue


        #     votos = {}
        #     for regla in reglasCoinciden:
        #         clase_pred = regla[atributos[-1]]
        #         votos[atributos[-1]] = votos.get(atributos[-1], 0) + 1 
        #     max_votos = max(votos.values())
            
        #     mejores = [clase for clase, v in votos.items() if v == max_votos]
        #     # Si hay empate se elige aleatoriamente
        #     pred = random.choice(mejores)

        #     if pred == x[atributos[-1]]:
        #         aciertos += 1
        
            

        # return aciertos / datos.shape[0]
            

