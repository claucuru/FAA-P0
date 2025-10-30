from Datos import Datos
from ClasificadorNaiveBayes import MultinomialNB
from Clasificador import Clasificador
from EstrategiaParticionado import ValidacionCruzada, ValidacionSimple
from ClasificadorKNN import KNN, DistanceMetricKNN
import numpy as np

if __name__ == '__main__':

    np.random.seed(42)

    naiveBayes = MultinomialNB()
    knn = KNN(5, DistanceMetricKNN.EUCLIDES)
    
    validacionCruzada = ValidacionCruzada(5)
    validacionSimple = ValidacionSimple(5, 0.3)

    dataset=Datos('../datasets/fuga_telefonia.csv')
    
    errores_vc = naiveBayes.limpiar_numpy(naiveBayes.validacion(validacionCruzada, dataset))
    print("Errores validacion cruzada: \n", errores_vc)
    print("Media validacion cruzada: ",np.mean(errores_vc))

    errores_vs = naiveBayes.limpiar_numpy(naiveBayes.validacion(validacionSimple, dataset))
    print("Errores validacion simple: \n", errores_vs)
    print("Media validacion  simple:",np.mean(errores_vs))

