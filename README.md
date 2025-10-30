# Práctica 0

**Autores:** Claudia Cuevas Ruano, Pablo Tejero Lascorz  
**Pareja:** 03

## NAIVE BAYES
### **Naive Bayes propio**
Se ha implementado un algoritmo que entrena y clasifica los datos dependiendo de si estos son continuos o no. En caso de serlo, se ha aplicado la expresión de la distribución normal. 

#### **Archivo fuga_telefonica.csv**
 
- Validación cruzada, 5 particiones:  
    - *Errores:*  Devuelve una media de error por partición  
 [0.30180180180180183, 0.3063063063063063, 0.34234234234234234, 0.32882882882882886, 0.30180180180180183]   
    - *Media errores*:  0.3162162162162162  

- Validación simple:
    - *Errores*:  0.35435435435435436  

- Validación cruzada, 5 particiones. **Sin correccion de Laplace**:  
    - *Errores:*  Devuelve una media de error por partición  
 [0.36036036036036034, 0.3963963963963964, 0.3063063063063063, 0.2747747747747748, 0.3108108108108108]
    - *Media errores*:  0.32972972972972975  

- Validación simple. **Sin correccion de Laplace**:
    - *Errores*:  0.0.32732732732732733  


#### **Archivo wdbc.csv**
 
- Validación cruzada, 5 particiones:  
    - *Errores:*  Devuelve una media de error por partición  
    [0.11403508771929824, 0.05263157894736842, 0.06140350877192982, 0.05263157894736842, 0.05309734513274336]
    - *Media errores*:  0.06675981990374165

- Validación simple:
    - *Errores*: 0.04678362573099415

- Validación cruzada, 5 particiones. **Sin corrección de Laplace**:  
    - *Errores:*  Devuelve una media de error por partición  
 [0.06140350877192982, 0.09649122807017543, 0.03508771929824561, 0.10526315789473684, 0.061946902654867256]
    - *Media errores*:  0.07203850333799099

- Validación simple. **Sin correccion de Laplace**:
    - *Errores*:  0.07017543859649122

### **Naive Bayes ScikitLearn**
Se han utilizado las funciones de la librería de Sklearn MultinomialNB y GaussianNB para los casos de validación cruzada y simple. Al usar MultinomialNB ha sido necesario estandarizar los datos para evitar datos continuos. 
#### **Archivo fuga_telefonica.csv**
- Validación cruzada, 5 particiones:
    - MultinomialNB:  
    *Errores:* 0.427
    - GaussianNB:
    *Errores*: 0.370
- Validación simple:
    - MultinomialNB:  
    *Errores:* 0.441
    - GaussianNB:
    *Errores*: 0.378

#### **Archivo wdbc.csv**
- Validación cruzada, 5 particiones:
    - MultinomialNB:  
    *Errores:* 0.165
    - GaussianNB:
    *Errores*: 0.061
- Validación simple:
    - MultinomialNB:  
    *Errores:* 0.105
    - GaussianNB:
    *Errores*: 0.064


Conclusiones
-
Contemplamos dos implementaciones distintas, ya que el NaiveBayes propio combina MultinomialNB y GaussianNB, usando uno u otro dependiendo del tipo de dato. 
Observamos que el NaiveBayes propio maneja datos mixtos de manera más efectiva que MultinomialNb de sklearn y que las diferencias entre validación simple y cruzada son moderadas, indicando estabilidad. 

La corrección de Laplace evita probabilidades cero en atributos nominales. Quitándolo, observamos un pequeño aumnto del error, el cual aumenta cuanto más mixto sea el dataset. 

