
## 2. Datos
Para el entrenamiento del modelo se usó la base de datos `The Mist Database` (http://yann.lecun.com/exdb/mnist/). Esta se encuentra divida en 60 mil imagénes de _training_ y 10 mil de _testing_. Para este proyecto se unira toda la base y se usará una propia división basado en el método `train_test_split` de `sklearn`.

Las imágenes son de 28x28 pixeles. Al ser transformadas en números, se vectorizan a 784 pixeles con valores entre 0-1. Cada columna será un pixel, y cada fila será una observación, de tal manera que el vector de explicativas es de 56 mil x 784.

<p align="center">
  <img src="figures/digits.png" width="700">
</p>


## 3. Metodología
(((Estructuras)))


## 4. Resultados
### 4.1 Selección de modelo
<p align="center">
  <img src="figures/model_selection_mse_stats.png" width="700">
</p>


### 4.2 Función de pérdida
<p align="center">
  <img src="figures/selected-loss.png" width="700">
</p>


### 4.3 Predicciones
<p align="center">
  <img src="figures/confmatrix_testing.png" width="700">
</p>

<p align="center">
  <img src="figures/model_selection_digits_hits.png" width="700">
</p>

<p align="center">
  <img src="figures/model_selection_digits_errors.png" width="700">
</p>


## 5. Conclusiones
1. Modelo de clasificación de imágenes cuenta con gran ajuste en los sets de train y test. El margen de error es muy pequeño.
2. Resultados robustos a diferentes valores iniciales, tomados al azar.

## 6. Agenda
1. Dibujo de dígito que no coincida con las dimensiones del entrenamiento para demostrar que el modelo no performa bien sin capas convulsionales.