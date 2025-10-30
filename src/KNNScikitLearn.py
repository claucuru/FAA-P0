from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

from Datos import Datos

if __name__ == '__main__':
    knn = KNeighborsClassifier(
        n_neighbors=5, 
        metric = 'euclidean',
        weights='uniform'
    )

    dataset=Datos('../datasets/fuga_telefonia.csv')


    # Obtenemos los atributos y las clases
    atributos = dataset.datos.iloc[:, :-1].values
    clases = dataset.datos.iloc[:, -1].values

    # Estandarizamos los atributos para KNN
    scaler = StandardScaler()
    atributos_estandarizados = scaler.fit_transform(atributos)

    # Obtenemos las particiones 
    dataset_train, dataset_test, class_train, class_test = train_test_split(
        atributos_estandarizados, clases, test_size=0.3, random_state=42, stratify=clases
    )

    # Calculamos el error con validacion simple
    knn.fit(dataset_train, class_train)
    aciertos = knn.score(dataset_test, class_test)
    print(f"KNN (k = 5), metric=euclidean \n\t Aciertos: {aciertos:.3f}  \n\t Errores: {1 - aciertos:.3f}")

    # Validación cruzada
    vc_predicciones = cross_val_score(knn, atributos_estandarizados, clases, cv=5)
    print(f"Validacion cruzada (5-fold): \n\t Media aciertos: {vc_predicciones.mean():.3f} \n\t Media errores {1- vc_predicciones.mean():.3f}\n\t Desviacion tipica: {vc_predicciones.std() *2:.3f}")