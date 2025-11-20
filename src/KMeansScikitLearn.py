from collections import Counter
import sys
from Datos import Datos
from ClasificadorRegLog import ClasificadorRegLog
from Estandarizador import Estandarizador2
from EstrategiaParticionado import ValidacionSimple
from sklearn.cluster import KMeans 
import numpy as np

def realizar_estudio (k, attrs_estandarizados, clases):

    print("="*60)
    print("K-MEANS (sklearn) CON K = {k}")
    print("="*60)
    
    kmeans = KMeans(n_clusters=k, max_iter=300, random_state=7)
    kmeans.fit(attrs_estandarizados.values)

    sse = kmeans.inertia_

    labels = kmeans.labels_

    #Los clusters deben ser diferentes
    unique_clusters = np.unique(labels)
    print(f"Se han encontrado un total de {len(unique_clusters)}")
    print(f"Los clusters van de {min(unique_clusters)} a {max(unique_clusters)}")
    print("Puntos por cluster:\n", Counter(labels))
    
    print("\n----Distribucion por cluster y clase:----\n")
    tabla = {}
    for c in range (k):
        indices = np.where(labels==c)[0]
        if len(indices) == 0:
            print(f"El cluster {c} esta vacio")
            tabla[c] = Counter()
            continue
        
        clases_cluster = clases.iloc[indices]
        tabla[c] = Counter(clases_cluster)

        #Pureza del cluster
        total = len(clases_cluster)
        if total > 0:
             clase_mayoritaria = tabla[c].most_common(1)[0] #Devuelve una lista con el elemento más frecuente y su conteo
             pureza = clase_mayoritaria[1] / total
             print(f"Cluster {c}: {dict(tabla[c])} Pureza: {pureza:.1%}")
        else:
            print(f"El cluster {c} esta vacio")
    
    # Mapeo y predicciones
    mapeo_cluster = {}
    for c in range(k):
        #Si un cluster esta vacio
        if len(tabla[c]) == 0:
            mapeo_cluster[c] = None
            continue
        #Obtenemos la clase más frecuente del cluster. 
        mapeo_cluster[c] = tabla[c].most_common(1)[0][0] #Devuelve una lista con el elemento más frecuente y su conteo

    # Asignamos a cada punto la clase mayoritaria de su cluster    
    predicciones = np.array([mapeo_cluster[c] for c in labels])

    # Matriz de confusión
    print("\n----Matriz de confusion----\n")
    matriz = np.zeros((10,10), dtype=int)
    #Llenamos la matriz, siendo la fila la clase real y la columna la predicción
    for clase_real, prediccion in zip(clases, predicciones):
        matriz[int(clase_real)][int(prediccion)] += 1
    
    for i in range(10):
        print(f"{i}: {matriz[i].tolist()}")

    #Calculamos la accuracy
    # Calculamos el porcentaje de predicciones correctas
    acc = np.mean(predicciones == clases.values)
    print(f"\nPorcentaje de pureza total: {acc:.2%}")
    print(f"SSE SKLEARN: ", sse)

    return acc

if __name__ == "__main__":

    dataset = Datos("../datasets/digitos.csv")

    dataset.convertir_a_float()

    attrs = dataset.datos.iloc[:, :-1]
    clases = dataset.datos.iloc[:, -1]

    attrs_estandarizados = Estandarizador2(dataset).estandarizarDatos(dataset.datos)
    
    resultados = {}
    for k in [8,10,12]:
        resultados[k] = realizar_estudio(k, attrs_estandarizados=attrs_estandarizados, clases=clases)
    
    print ("\n" + "="*60)
    print("RESULTADOS FINALES CON SKLEARN")
    print ("="*60)
    for k, acc in resultados.items():
        print(f"K = {k:2d} -> Pureza: {acc:.2%}")



