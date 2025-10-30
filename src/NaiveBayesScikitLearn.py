from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

from Datos import Datos

if __name__ == '__main__':
    np.random.seed(42)
    dataset=Datos('../datasets/wdbc.csv')

    # Obtenemos los atributos y las clases
    atributos = dataset.datos.iloc[:, :-1].values
    clases = dataset.datos.iloc[:, -1].values

    encoder = LabelEncoder()
    clases_codificadas = encoder.fit_transform(clases)

    # Obtenemos las particiones 
    dataset_train, dataset_test, class_train, class_test = train_test_split(
        atributos, clases_codificadas, test_size=0.3, random_state=42, stratify=clases
    )

    # Validacion simple
    mnb = MultinomialNB()
    mnb.fit(dataset_train, class_train)
    predicciones_mnb = mnb.predict(dataset_test)

    aciertos = accuracy_score(class_test, predicciones_mnb)
    print(f"Validacion simple MultinomialNB: \n\t Aciertos: {aciertos:.3f} \n\t Errores: {1 - aciertos:.3f}")

    gnb = GaussianNB()
    
    gnb.fit(dataset_train, class_train)
    predicciones_gnb = gnb.predict(dataset_test)

    aciertos_gnb = accuracy_score(class_test, predicciones_gnb)
    print(f"Validacion simple GaussianNB: \n\t Aciertos: {aciertos_gnb:.3f} \n\t Errores: {1 - aciertos_gnb:.3f}")

    # Validación cruzada
    scaler = MinMaxScaler()
    atributos_estandarizados = scaler.fit_transform(atributos)
    predicciones_mnb = cross_val_score(MultinomialNB(), atributos_estandarizados, clases_codificadas, cv=5)
    print(f"Validacion cruzada  MultinomialNB (5-fold): \n\t Media de aciertos: {np.mean(predicciones_mnb):.3f} \n\t Media de errores: {1 - np.mean(predicciones_mnb):.3f}")

    predicciones_gnb = cross_val_score(GaussianNB(), atributos, clases_codificadas, cv=5)
    print(f"Validacion cruzada GaussianNB (5-fold): \n\t Media de aciertos: {np.mean(predicciones_gnb)} \n\t Media de errores: {1 - np.mean(predicciones_gnb):.3f}")
