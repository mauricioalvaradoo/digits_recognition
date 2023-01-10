# Hand-Written Digits Recognition
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from functions import modelos_dr


# Estimación ===============================================================
X_train = pd.read_pickle("./data/X_train.pkl")
y_train = pd.read_pickle("./data/y_train.pkl")
X_test = pd.read_pickle("./data/X_test.pkl")
y_test = pd.read_pickle("./data/y_test.pkl")

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)


"""
# Selección de modelo ======================================================
Se proponen tres arquitecturas neuronales diferentes:

Modelo_1:
    * Layer0 = Insumos
    * Layer1 = 40 neuronas
    * Layer2 = 20 neuronas
    * Layer3 = 10 neuronas

Modelo_2:
    * Layer0 = Insumos
    * Layer1 = 60 neuronas
    * Layer2 = 30 neuronas
    * Layer3 = 20 neuronas
    * Layer4 = 10 neuronas  

Modelo_3:
    * Layer0 = Insumos
    * Layer1 = 20 neuronas
    * Layer2 = 20 neuronas
    * Layer3 = 15 neuronas
    * Layer4 = 15 neuronas 
    * Layer5 = 10 neuronas

Se comparán los resultados mediante MSE, debido a que el layer final es lineal.
Para todos se estimará con 200 epochs y 5 veces. El MSE final será el promedio
de las 5 estimaciones.
"""


iteration_error = []

for i in range(1, 5):
    
    train_error = []

    modelos = modelos_dr.model_selection_list()
    
    for model in modelos:
        
        model.compile(
                loss = SparseCategoricalCrossentropy(from_logits=True),
                optimizer = Adam(0.01),
            )

        print(f"Entrenando {model.name}. Iteración #{i}...")

        model.fit(
            X_train, y_train,
            epochs = 200,
            verbose = False
        )
        
        print("Finalizado!\n")
        
        # Creación del threshold
        threshold = 0.5
        yhat = model.predict(X_train)
        yhat = tf.math.sigmoid(yhat)
        yhat = np.where(yhat >= threshold, 1, 0)
        train_error = np.mean(yhat != y_train)
        train_error.append(train_error)

    iteration_error.append({i: train_error})


# for j in range(len(train_error)):
#     print(
#         f"Modelo #{j+1}: MSE de entrenamiento: {train_error[j]:.5f}"
#     )



# Estimación del modelo seleccionado =======================================
models, results, fx = modelos_dr.fit_selected(X_train, y_train, epochs=50, iters=100) # 2 horas aprox

df = pd.DataFrame(results).T
plt.plot(df)
plt.show()



# Función de pérdida =======================================================
plt.figure(figsize=(6,4))

plt.plot(results.history["loss"], color="black") 
# https://stackoverflow.com/questions/62186595/standard-deviation-using-numpy

plt.xlabel("# Iteraciones")
plt.ylabel("Magnitud")
plt.title("Magnitud de la función de pérdida por iteraciones")
plt.savefig("./figures/log-loss.pdf")
plt.show()


# Gráfico label, predict


# Gráfico de observaciones con errores




