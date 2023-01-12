import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from functions import model_selection, table


# Data =====================================================================
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
Para todos se estimará con 100 epochs y 10 veces. El MSE final será el promedio
de las 10 estimaciones.
"""


iteration_error = []

for i in range(1, 11):
    
    train_error = []
    modelos = model_selection.selection_list()
    
    for model in modelos:
        
        model.compile(
                loss = SparseCategoricalCrossentropy(from_logits=True),
                optimizer = Adam(0.001),
            )  
        print(f"Entrenando {model.name}. Iteración #{i}...")

        model.fit(
            X_train, y_train,
            epochs = 100,
            verbose = False
        )
        
        # Selección del label con mayor probabilidad
        logits = model.predict(X_test)
        fx = tf.nn.softmax(logits)
        
        yhat = []
        for j in range(len(fx)):
            yhat.append(np.argmax(fx[j]))
        
        yhat = np.array([np.asarray(yhat)]).T
        error = np.mean(yhat != y_test) # % de errores/total
        train_error.append(error)

    iteration_error.append({i: train_error})


# Recuperando los MSE
mse_m1 = []; mse_m2 = []; mse_m3 = []

for i in iteration_error:
    i = list(i.values())[0]
    mse_m1.append(i[0])
    mse_m2.append(i[1])
    mse_m3.append(i[2])



# MSE modelos e iteraciones ================================================
df_error = pd.DataFrame({"Modelo 1": mse_m1, "Modelo 2": mse_m2, "Modelo 3": mse_m3})
df_error.index.name = "Iteraciones"
df_error.to_csv("./results/model_selection_mse.csv")
df_error

df_error = np.round(df_error, 4)
fig, ax = table.to_figure(df_error.reset_index(), header_columns=0, col_width=2.0)
fig.savefig("./figures/model_selection_mse.png")


# MSE modelos y estadísticas =============================================== 
df_error_stats = pd.DataFrame({"Modelos": ["Modelo 1", "Modelo 2", "Modelo 3"], "Mediana": df_error.median(), "Std": df_error.std()})
df_error_stats.set_index("Modelos", inplace=True)
df_error_stats.to_csv("./results/model_selection_mse_stats.csv")
df_error_stats

df_error_stats = np.round(df_error_stats, 4)
fig, ax = table.to_figure(df_error_stats.reset_index(), header_columns=0, col_width=2.0)
fig.savefig("./figures/model_selection_mse_stats.png")




