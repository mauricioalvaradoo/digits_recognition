import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import utils


# Data =====================================================================
X_train = pd.read_pickle("./data/X_train.pkl")
y_train = pd.read_pickle("./data/y_train.pkl")
X_test = pd.read_pickle("./data/X_test.pkl")
y_test = pd.read_pickle("./data/y_test.pkl")

X_train = np.asarray(X_train).reshape(56_000, 28, 28)
y_train = np.asarray(y_train).reshape(56_000,)
X_test = np.asarray(X_test).reshape(14_000, 28, 28)
y_test = np.asarray(y_test).reshape(14_000,)


"""
# Selección de modelo ======================================================
Se proponen tres arquitecturas neuronales diferentes:

Modelo_1:
    * Convolucional = 32 capas
    * Pooling = 2x2
    * Convolucional = 64 capas
    * Pooling = 2x2
    * Layer1 = 40 neuronas
    * Layer2 = 20 neuronas
    * Layer3 = 10 neuronas

Modelo_2:
    * Convolucional = 32 capas
    * Pooling = 2x2
    * Convolucional = 64 capas
    * Pooling = 2x2
    * Layer1 = 60 neuronas
    * Layer2 = 30 neuronas
    * Layer3 = 20 neuronas
    * Layer4 = 10 neuronas  

Modelo_3:
    * Convolucional = 32 capas
    * Pooling = 2x2
    * Convolucional = 64 capas
    * Pooling = 2x2
    * Layer1 = 20 neuronas
    * Layer2 = 20 neuronas
    * Layer3 = 15 neuronas
    * Layer4 = 15 neuronas 
    * Layer5 = 10 neuronas

Todos consideran una dos capas convolucionales previas de 32 y 64 núcleos,
respectivamente. La capa de agrupación asociada a cada capa convolucional es de
agrupaciones de (2, 2).
Se comparán los resultados mediante MSE, debido a que el layer final es lineal.
Para todos se estimará con 100 epochs y 10 veces. El MSE final será el promedio
de las 10 estimaciones.
"""

# Callbacks -> Frenar estimación cuando se alcance MSE (0.01) o ajuste (99.5%)
class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if ((logs.get("loss")<0.001) or (logs.get("accuracy")>0.995)):
            self.model.stop_training = True
callbacks = myCallback()


# Model Selection
iteration_error = []
iteration_accur = []

for i in range(1, 11):
    
    train_error = []
    train_accur = []
    modelos = utils.selection_list()
    
    for model in modelos:
        
        model.compile(
                loss = SparseCategoricalCrossentropy(from_logits=True),
                optimizer = Adam(0.001),
                metrics = ["mean_squared_error", "accuracy"]
        )  
        print(f"Entrenando {model.name}. Iteración #{i}...")

        history = model.fit(
            X_train, y_train,
            epochs = 100,
            verbose = False,
            callbacks = [callbacks]
        )
        
        # Selección del label con mayor probabilidad
        logits = model.predict(X_test)
        fx = tf.nn.softmax(logits)
        
        yhat = []
        for j in range(len(fx)):
            yhat.append(np.argmax(fx[j]))
        
        # Porcentaje Error/Total
        yhat = np.array([np.asarray(yhat)])
        error = np.mean(yhat != y_test)
        train_error.append(error)
        
        # Ajuste
        train_accur.append(history.history["accuracy"][-1])
        
        
    iteration_error.append({i: train_error})
    iteration_accur.append({i: train_accur})


# Recuperando los MSE
mse_m1 = []; mse_m2 = []; mse_m3 = []

for i in iteration_error:
    i = list(i.values())[0]
    mse_m1.append(i[0])
    mse_m2.append(i[1])
    mse_m3.append(i[2])

# Recuperando los ajustes
acc_m1 = []; acc_m2 = []; acc_m3 = []

for i in iteration_accur:
    i = list(i.values())[0]
    acc_m1.append(i[0])
    acc_m2.append(i[1])
    acc_m3.append(i[2])



# MSE modelos e iteraciones ===================================================
df_error = pd.DataFrame({"Modelo 1": mse_m1, "Modelo 2": mse_m2, "Modelo 3": mse_m3})
df_error.index.name = "Iteraciones"
df_error.to_csv("./results/model_selection_mse.csv")
df_error

df_error = np.round(df_error, 4)
fig, ax = utils.to_figure(df_error.reset_index(), header_columns=0, col_width=2.0)
fig.savefig("./figures/model_selection_mse.png")


# Ajuste modelos e iteraciones ================================================
df_accu = pd.DataFrame({"Modelo 1": acc_m1, "Modelo 2": acc_m2, "Modelo 3": acc_m3})
df_accu.index.name = "Iteraciones"
df_accu.to_csv("./results/model_selection_acc.csv")
df_accu

df_accu = np.round(df_accu, 4)
fig, ax = utils.to_figure(df_accu.reset_index(), header_columns=0, col_width=2.0)
fig.savefig("./figures/model_selection_acc.png")


# MSE y ajuste modelos y estadísticas =========================================
df_stats = pd.DataFrame({
    "Modelos": ["Modelo 1", "Modelo 2", "Modelo 3"],
     "Error": df_error.median(),
     r"Std error": df_error.std(),
     "Ajuste": df_accu.median(),
     r"Std ajuste": df_accu.std(),
})

df_stats.set_index("Modelos", inplace=True)
df_stats.to_csv("./results/model_selection_stats.csv")
df_stats

df_stats = np.round(df_stats, 4)
fig, ax = utils.to_figure(df_stats.reset_index(), header_columns=0, col_width=2.0)
fig.savefig("./figures/model_selection_stats.png")




