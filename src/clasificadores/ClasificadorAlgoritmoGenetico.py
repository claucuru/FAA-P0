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

        self.best_priori = None
        self.best_fitness = -np.inf
        self.best_individuo = None

        self.encoder = OneHotEncoder(sparse_output=False)
        self.num_val_por_atributo = None
        self.n_attrs = 0


    def entrenamiento(self, datos: Datos):
        attrs = datos.datos.iloc[:,:-1]
        clase = datos.datos.iloc[:,-1]

        self.n_attrs = attrs.shape[1]

        # Ajustar encoder a los datos de entrenamiento (solo atributos, no clase)
        self.encoder.fit(attrs)

        self.num_val_por_atributo = [
            len(valores_atributo) for valores_atributo in self.encoder.categories_
        ]

        # Transformar los datos
        datosTrain = np.ndarray(shape=(datos.datos.shape[0], sum(self.num_val_por_atributo) + 1))
        datosTrain[:,:-1] = self.encoder.transform(attrs)
        datosTrain[:,-1]  = clase

        len_regla = datosTrain.shape[1]

        # Obtener la clase con mejor priori
        count_clase_1 = np.sum(datosTrain[:,-1])
        count_clase_0 = datosTrain.shape[1] - count_clase_1

        if count_clase_0 > count_clase_1:
            self.best_priori = 0
        else:
            self.best_priori = 1

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

        for i in range(self.n_individuos):
            n_reglas = random.randint(1, self.max_reglas)

            individuo = np.random.randint(0, 2, (n_reglas, len_regla))

            poblacion.append(individuo)

        print("POBLACION INICIAL")
        print("-"*40)
        print(poblacion)


        for _ in range(self.max_epochs):
            # Seleccion de progenitores para la descendencia. Go spin the wheel.
            # Recomendado: https://es.piliapp.com/random/wheel/
            # Aplicamos la funcion F a cada individuo de la muestra. Dividimos el
            # resultado entre la suma de las funciones F de cada individuo.
            fitness_values = np.array([self._fitness(individuo, datosTrain) for individuo in poblacion])
            fitness_probs = fitness_values / np.sum(fitness_values)
            fitness_cum_probs = np.cumsum(fitness_probs)

            progenitores_seleccionados = [
                poblacion[np.searchsorted(fitness_cum_probs, random.uniform(0, 1))]
                for _ in range(self.n_individuos)
            ]
            np.random.shuffle(progenitores_seleccionados)
         

            # Lista de descendientes
            descendientes = []

            # Arcane s2ep9
            for i in range(0, len(progenitores_seleccionados), 2):
                # Progenitores para el cruce
                p1: np.ndarray = progenitores_seleccionados[i]
                p2: np.ndarray = progenitores_seleccionados[(i + 1) % self.n_individuos]

                # Se escoge una regla al azar de cada progenitor y un punto de
                # cruce aleatorio.
                idx_r1 = random.randint(0, p1.shape[0] - 1)
                idx_r2 = random.randint(0, p2.shape[0] - 1)
                r1 = p1[idx_r1]
                r2 = p2[idx_r2]

                punto_cruce = random.randint(1, len_regla - 1)

                # Formacion de sucesores
                nr1 = np.concatenate( (r1[:punto_cruce], r2[punto_cruce:]) )
                nr2 = np.concatenate( (r1[punto_cruce:], r2[:punto_cruce]) )

                s1 = np.copy(p1); s1[idx_r1] = nr1
                s2 = np.copy(p2); s2[idx_r2] = nr2

                descendientes.append(s1)
                descendientes.append(s2)

            # Mutacion: cada bit tiene una probabilidad de mutar
            aux = len_regla * len(poblacion)

            for individuo in progenitores_seleccionados:
                individuo_mutado = np.copy(individuo)
                mutado = False

                for regla in individuo:
                    for i, bit in enumerate(regla):
                        prob_mutacion = random.randint(1, aux)

                        if prob_mutacion == 1:
                            regla[i] = (~bit) & 1
                            mutado = True

                if mutado:
                    descendientes.append(individuo_mutado)

            #print("DESCENDIENTES")
            #print("-"*40)
            # print(descendientes)


            # Seleccion de supervivientes. La nueva generacion se convierte en la
            # nueva poblacion. Sin embargo, aplicamos un cierto grado de elitismo,
            # manteniendo a los k mejores individuos escogidos entre la poblacion
            # antigua y los sucesores generados.
            # len_pobl = len(poblacion)

            # k_mejores = int(len_pobl * self.elitismo)

            # np.random.shuffle(descendientes)
            # rand_descendientes = descendientes[k_mejores:self.n_individuos]

            # fitness_values = np.concatenate((fitness_values, np.array([
            #     self._fitness(individuo, datosTrain)
            #     for individuo in descendientes
            # ])))
            # idx_k_mejores = np.argsort(fitness_values)[-k_mejores:]

            # poblacion_k_mejores = [
            #     descendientes[i - len_pobl] if i >= len_pobl else poblacion[i]
            #     for i in idx_k_mejores
            # ]
            # poblacion_rand = [
                
            # ]

            # poblacion = rand_descendientes + poblacion_k_mejores

            # print("NUEVA POBLACION - BEST")
            # print("-"*40)
            # #print(poblacion)
            # print(poblacion_k_mejores[-1])
            # print("Fitness:", np.max(fitness_values))

            poblacion_combinada = poblacion + descendientes

            fitness_all = np.concatenate((
                fitness_values,
                np.array([
                    self._fitness(individuo, datosTrain)
                    for individuo in descendientes
                ])
            ))
            idx_sorted = np.argsort(fitness_all)

            k = int(self.elitismo * self.n_individuos)


            nueva_poblacion = [None for _ in range(self.n_individuos)]

            # Elegir k mejores
            for i, idx in enumerate(idx_sorted[:k]):
                nueva_poblacion[i] = poblacion_combinada[idx]

            # Descartar de la poblacion combinada los k mejores ya seleccionados y
            # escoger aleatoriamente el resto de individuos de la nueva poblacion
            idx_set = set(idx_sorted[:k])
            poblacion_combinada = [
                ind for i, ind in enumerate(poblacion_combinada) if i not in idx_set
            ]

            random.shuffle(poblacion_combinada)
            nueva_poblacion[k:] = poblacion_combinada[:self.n_individuos - k]

            print("NUEVA POBLACION - BEST")
            print("-"*40)
            poblacion = nueva_poblacion

            print(poblacion)


        # Seleccionar mejor individuo
        for individuo in poblacion:
            fitness = self._fitness(individuo, datosTrain)
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individuo = individuo

        print(self.best_individuo)
        print(self.best_fitness)


    def clasifica(self, datos: Datos):
        datosTest = self.encoder.transform(datos.datos.iloc[:,:self.n_attrs])
        predicciones = np.array([
            self._predecir_clase_individuo(self.best_individuo, muestra)
            for muestra in datosTest
        ])

        return predicciones


    def _fitness(self, individuo, datos: np.ndarray):
        # Cuantas instancias del dataset cumplen al menos una regla del individuo
        num_aciertos = 0

        # Contabilizar aciertos y errores para cada fila del dataset
        for muestra in datos:
            pred = self._predecir_clase_individuo(individuo, muestra)

            if pred == muestra[-1]:
                num_aciertos += 1

        return num_aciertos / datos.shape[0]


    def _predecir_clase_individuo(self, individuo: np.ndarray, muestra: np.ndarray):
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
                        break

                if cumple_regla:
                    coincidencias[regla[-1]] += 1


            if coincidencias[0] == coincidencias[1]:
                # Ninguna regla coincide con la muestra
                if coincidencias[0] == 0:
                    return None
                # Empate entre las dos clases
                else:
                    pred = self.best_priori
            else:
                if coincidencias[0] > coincidencias[1]:
                    pred = 0
                else:
                    pred = 1

            return pred
