import sys
import matplotlib.pyplot as plt
from Datos import Datos
from ClasificadorRegLog import ClasificadorRegLog
from Estandarizador import Estandarizador2
from EstrategiaParticionado import ValidacionSimple


if __name__ == "__main__":
    use_dataset = sys.argv[1]

    if use_dataset == 'w':
        dataset = Datos("./datasets/wdbc.csv")
    elif use_dataset == 't':
        dataset = Datos("./datasets/fuga_telefonia.csv")


    print("Sin estandarizar:", dataset.datos.shape)
    dataset.convertir_a_float()
    dataset.datos.iloc[:,:-1] = Estandarizador2(dataset).estandarizarDatos(dataset.datos)
    print("Con estandarizado:", dataset.datos.shape)


    validador = ValidacionSimple(1, 0.2)
    validador.creaParticiones(dataset, 67)
    datosTrain: Datos = dataset.particion(validador.particiones[0].indicesTrain)
    datosTest: Datos = dataset.particion(validador.particiones[0].indicesTest)


    clasificador = ClasificadorRegLog(1, 400)
    clasificador.entrenamiento(datosTrain)
    predicciones = clasificador.clasifica(datosTest)
    vp, fp, fn, vn = clasificador.analisis_roc(datosTest, predicciones)

    print("VP:", vp)
    print("FP:", fp)
    print("FN:", fn)
    print("VN:", vn)

    vpr = vp / (vp + fn)
    fpr = fp / (fp + vn)
    fnr = fn / (vp + fn)
    vnr = vn / (fp + vn)

    print(f"VP Ratio: {vpr * 100:.4f}%")
    print(f"FP Ratio: {fpr * 100:.4f}%")
    print(f"FN Ratio: {fnr * 100:.4f}%")
    print(f"VN Ratio: {vnr * 100:.4f}%")

    print("Aciertos:", vp + vn, "/", vp + vn + fn + fp, f"({(vp + vn) / (vp + vn + fn + fp)* 100:.4f} %)")

    x = (0.12, 0.29, 0.50)
    y = (0.80, 0.60, 0.50)
    x = (fpr)
    y = (vpr)

    ticks = [ n / 10 for n in range(10 + 1) ]
    labels = [ str(n) for n in ticks ]

    plt.figure()
    plt.grid(True)
    plt.plot((0, 1), (0, 1), "--", color="red")
    plt.plot(x, y, "o", color="blue")
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.tight_layout()
    plt.show()
