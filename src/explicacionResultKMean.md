# K-MEANS
## K-MEANS PROPIO
El algoritmo de K-MEANS se ha implementando siguiendo los siguientes pasos:
1. Elegir el numero de clusters k. Aunque se pueden utilizar herramientas como el método del codo, se han elegido aleatoriamente
 2. Seleccionar aleatoriamente los centroides, que indican dónde está el centro de cada clúster
 3. Se asigna cada punto del cluster al centroide más cercano usando la distancia euclidea 
 4. Se evalua la calidad de los clústers (se usa función objetivo)
 5. Se recalculan los centroides, dándoles como valor el punto medio entre los datos de cada grupo
 6. Se repiten los pasos 3 y 4 hasta que el algoritmo converga según los pasos establecidos por nosotros:  
    Híper parámetros:   
    6.1 Número de clústers: en cuántos grupos vamos a separar los datos
    6.2 Número de iteraciones máximas, como criterio de convergencia del algoritmo  
    6.3 Límite de la métrica de evaluación, otro criterio de convergencia


## K-MEANS CON SKLEARN
Se ha utilizado la librería de sklearn, y se ha calculado la matriz de confusión al igual que en el caso anterior, así como la distribución por cluster y clase. 



## **Cuestiones**

#### **1. Para K=10, comprobar si se puede asignar de forma unívoca cada clúster a un dígito atendiendo a la clase mayoritaria de los patrones agrupados por clúster**  
No, no del todo. Para dígitos como 0, 6, 4 y 7 hay un cluster dedicado a cada uno y la pureza obtenida es muy alta. Sin embargo, para los dígitos 8, 3 y 9 y 11 no sucede lo mismo, pues por ejemplo, el 8 se reparte entre clusters 3, 5, 7 y 8. Son clusters muy mezclados y hacen difícil una asignación unívoca.

#### **2. Analizar qué tipos de dígitos se identifican más fácilmente y cuales se confunden entre ellos. Para ello se puede hacer uso de la matriz de confusión multiclase a partir del número de puntos en cada clase y sabiendo que hay entre 170 y 185  ejemplos de cada dígito**  
- Dígitos muy bien agrupados: 0, 4, 6 y 7. Son de los trazos más simples
- Dígitos medianamente bien identificados: 
    - 1: Aparece mezclado con 7, 8 y 4
    - 2: Se divide en varios clusters
- Dígitos difíciles de separar:
    - 3: Se confunde con 5, 8 y 9
    - 5: Mezclado con 8, 3, 2 y 9
    - 8: Aparece en casi todos los clusters
    - 9: Se mezcla con 3 y 8

#### **3.Probar con otros valores de K=[8,12] y comparar los resultados con los obtenidos para K=10. ¿Cuál sería el mejor K para este problema?**  

**Pureza**                                       

|              | K = 8   | K = 10 | K = 12 | 
|--------------|---------|--------|--------|
| PROPIO       | 54.31%  | 66.17% | 74.07% |
| SKLEARN      | 59.77%  | 59.65% | 67.84% |

**SSE**
|              | K = 8    |  K = 10  | K = 12    |
|--------------|----------|----------|-----------|
| PROPIO       | 74566.85 | 69810.90 | 66159.067 |
| SKLEARN      | 75200.909| 70336.97 | 65966.26  |

### K = 8
Sklearn produce clusters más alineados con las clases, mientras que el algoritmo propio produce clusters más compactos pero peor definidos con respecto a las clases reales. Son muy pocos clusters para 10 dígitos y están muy mezclados. 

### K = 10
El algoritmo propio funciona mejor que el sklearn para este dataset. Produce mejor pureza y mejor SSE. Por otro lado, sklearn empeora ligeramente respecto a K=8

### K = 12
El algoritmo propio obtiene una mayor pureza, aunque el SSE es levemente peor que el obtenido con sklearn. Es el que mejor pureza tiene en ambas implementaciones y permite clusters especializados para variantes. Además, solo hay un dígito problemático.

En conclusión, K=12 es claramente el mejor K ya que obtiene la mejor pureza, la mejor distribución de dígitos y los clústers más puros. 




