import sys
from Datos import Datos
from ClasificadorRegLog import ClasificadorRegLog
from Estandarizador import Estandarizador2
from EstrategiaParticionado import ValidacionSimple


if __name__ == "__main__":
    use_dataset = sys.argv[1]

    if use_dataset == 'w':
        dataset = Datos("../datasets/wdbc.csv")
    elif use_dataset == 't':
        dataset = Datos("../datasets/fuga_telefonia.csv")


    dataset.datos = dataset.datos[:10]

    dataset.convertir_a_float()
    print(dataset.datos.dtypes)

    estandarizador = Estandarizador2(dataset)
    dataset.datos.iloc[:,:-1] = estandarizador.estandarizarDatos(dataset.datos)
    print("Con estandarizado:", dataset.datos.shape)
    print(dataset.datos)
