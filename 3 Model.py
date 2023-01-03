# Hand-Written Digits Recognition

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential, Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)


# Data
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

model = Sequential(
    [
        Input(shape = (784,)),
        Dense(units = 50, activation = "sigmoid", name = "Layer1"),
        Dense(units = 20, activation = "sigmoid", name = "Layer2"),
        Dense(units = 1,  activation = "sigmoid", name = "Layer3")

    ], name = "Modelo1"
)

# model.summary()


model.compile(
    loss = BinaryCrossentropy(),
    optimizer = Adam(0.001),
)

model.fit(
    X_train, y_train,
    epochs = 20,
    verbose = False
)

# GRAFICO DE PERDIDA!
# GUARDAR MODELO
