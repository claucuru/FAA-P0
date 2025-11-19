import sys
from sklearn.linear_model import LogisticRegression
from EstrategiaParticionado import ValidacionSimple
from Datos import Datos


if __name__ == "__main__":
    use_dataset = sys.argv[1]

    if use_dataset == 'w':
        dataset = Datos("./datasets/wdbc.csv")
    elif use_dataset == 't':
        dataset = Datos("./datasets/fuga_telefonia.csv")


    validador = ValidacionSimple(1, 0.2)
    validador.creaParticiones(dataset, 67)
    particion = validador.particiones[0]
    datosTrain: Datos = dataset.particion(particion.indicesTrain)
    datosTest: Datos = dataset.particion(particion.indicesTest)

    classifier = LogisticRegression(random_state=67, max_iter=400).fit(
        datosTrain.datos.iloc[:,:-1], datosTrain.datos.iloc[:,-1]
    )
    classifier.predict(datosTest.datos.iloc[:,:-1])


    # X, y = load_iris(return_X_y=True)
    # clf = LogisticRegression(random_state=0).fit(X, y)
    # clf.predict(X[:2, :])
    # array([0, 0])
    # clf.predict_proba(X[:2, :])
    # array([[9.82e-01, 1.82e-02, 1.44e-08],
    #     [9.72e-01, 2.82e-02, 3.02e-08]])
    # clf.score(X, y)