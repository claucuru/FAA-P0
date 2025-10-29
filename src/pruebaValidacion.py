from Datos import Datos
from ClasificadorNaiveBayes import MultinomialNB
from Clasificador import Clasificador
from EstrategiaParticionado import ValidacionCruzada, ValidacionSimple
from ClasificadorKNN import KNN, DistanceMetricKNN

if __name__ == '__main__':
    dataset_test=Datos('../datasets/heart-test.csv')
    
    dataset_train=Datos('../datasets/heart-train.csv')
    # estrategia=EstrategiaParticionado.ValidacionSimple(5, 0.8)

    print(dataset_test.nominalAtributos)
    print("Diccionario:")
    print(dataset_test.diccionario)
    print("\nDatos:")
    print(dataset_test.datos)
    print("Extraer datos: \n", dataset_test.extraeDatos(1))

    nb = MultinomialNB()

    nb.entrenamiento(dataset_test)
    modelo_limpio = nb.limpiar_numpy(nb.modelo)
    condicionales_limpios = nb.limpiar_numpy(nb.modelo["condicionales"])

    for clase, attrs in modelo_limpio["condicionales"].items():
        print(f"\nClase {clase:}")
        for i, valores in attrs.items():
            print(f"    Atributo {i}: {valores}")


    

    # print(json.dumps(condicionales_limpios[0][1], indent=2))
    # print("Prioris:")
    # print(modelo_limpio["prioris"])
    # print("\nCondicionales:")
    # print(modelo_limpio["condicionales"])
    predicciones = nb.clasifica(dataset_test)
    print("\nPredicciones:")
    print(predicciones)

    
    ###################################################### Validación
    print("="*10)
    naiveBayes = MultinomialNB()
    knn = KNN(5, DistanceMetricKNN.EUCLIDES)
    
    validacionSimple = ValidacionCruzada(4)

    dataset=Datos('../datasets/fuga_telefonia.csv')
    
    print("Errores:")
    print(knn.validacion(validacionSimple, dataset))

