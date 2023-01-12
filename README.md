# Reconocimiento de dígitos mediante Redes Neuronales
El proyecto fue desarrollado por Mauricio Alvarado y Marco Virú.

## 1. Objetivo
El objetivo es desarrollar un modelo mediante redes neuronales para la clasificación de dígitos. Para conseguirlo se requiere la transformación a formato numérico de imágenes de número entre el 0 y el 9.


## 2. Datos
Para el entrenamiento del modelo se usó la base de datos `The Mist Database` (http://yann.lecun.com/exdb/mnist/). Esta se encuentra divida en 60 mil imagénes de _training_ y 10 mil de _testing_. Para este proyecto se unira toda la base y se usará una propia división basado en el método `train_test_split` de `sklearn`.

Las imágenes son de 28x28 pixeles. Al ser transformadas en números, se vectorizan a 784 pixeles con valores entre 0-1. Cada columna será un pixel, y cada fila será una observación, de tal manera que el vector de explicativas es de 56 mil x 784.


(((Imagen de graficos)))


## 3. Metodología
(((Estructuras)))


## 4. Resultados
### 4.1 Selección de modelo
((((Modelo seleccionado))))


### 4.2 Función de pérdida
(((Bandas de confianza del modelo seleccionado))) 


### 4.3 Predicciones
((((Tabla de aciertos y desaciertos del modelo seleccionado y mediana))))

(((Graficos de aciertos y labels del modelo seleccionado y mediana)))
(((Graficos de desaciertos y labels del modelo seleccionado y mediana)))


### 5. Ejercicios de robustez
(((10 iteraciones para cada ejercicio con mediana y std)))
(((comparada con el modelo base)))


### 6. Página web -> Modelo guardado
(((Modelo base y mediana a web)))


