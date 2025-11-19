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


    print("Sin estandarizar:", dataset.datos.shape)
    dataset.convertir_a_float()
    dataset.datos.iloc[:,:-1] = Estandarizador2(dataset).estandarizarDatos(dataset.datos)
    print("Con estandarizado:", dataset.datos.shape)

    clasificador = ClasificadorRegLog(1, 400)

    errores = clasificador.validacion(ValidacionSimple(1, 0.2), dataset, 67)
    print(f"{errores=}")