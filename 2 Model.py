# Hand-Written Digits Recognition
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from numba import njit

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Data =====================================================================
X_train = pd.read_pickle("./data/X_train.pkl")
y_train = pd.read_pickle("./data/y_train.pkl")
X_test = pd.read_pickle("./data/X_test.pkl")
y_test = pd.read_pickle("./data/y_test.pkl")

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)


"""""
## Modelo
Se usará una arquitectura neuronal que consiste en lo siguiente:

* Layer0 = Insumos
* Layer1 = 50 neuronas
* Layer2 = 20 neuronas
* Layer3 = 1 neuronas

"""""


tf.random.set_seed(19)

model = Sequential(
    [
        Input(shape = (784,)),
        Dense(units = 50, activation = "sigmoid", name = "Layer1"),
        Dense(units = 20, activation = "sigmoid", name = "Layer2"),
        Dense(units = 10,  activation = "linear", name = "Layer3") # En vez de "softmax"

    ], name = "Modelo1"
)

# model.summary()

model.compile(
    loss = SparseCategoricalCrossentropy(from_logits=True),
    optimizer = Adam(0.01),
)


results = model.fit(
    X_train, y_train,
    epochs = 500,
    verbose = False
) ### Que epochs message aparezca 10 veces


logits = model.predict(X_train)
fx = tf.nn.softmax(logits)


# Check historia -> parametros


# Función de pérdida =======================================================
plt.figure(figsize=(6,4))

plt.plot(results.history["loss"], color="black")

plt.xlabel("# Iteraciones")
plt.ylabel("Magnitud")
plt.title("Magnitud de la función de pérdida por iteraciones")

plt.savefig("./figures/log-loss.pdf")
plt.show()


# Cross-validation =========================================================





# Guardar modelo ===========================================================




