# Hand-Written Digits Recognition
import numpy as np
import pandas as pd
from numba import njit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
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
tf.random.set_seed(19)
model = Sequential(
    [
        Input(shape = (784,)),
        Dense(units = 50, activation = "sigmoid", name = "Layer1"),
        Dense(units = 20, activation = "sigmoid", name = "Layer2"),
        Dense(units = 10,  activation = "softmax", name = "Layer3")

    ], name = "Modelo1"
)

# model.summary()


model.compile(
    loss = SparseCategoricalCrossentropy(),
    optimizer = Adam(0.01),
)


results = model.fit(
    X_train, y_train,
    epochs = 100,
    verbose = False
)

# GRAFICO DE PERDIDA!
import matplotlib.pyplot as plt
plt.xlabel("#Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(results.history["loss"])

#MATRIZ DE CONFUSION
y_hat=results.predict(X_test)

y_hat = np.where(y_hat > 0.5, 1, 0)
conf = confusion_matrix(y_test, y_hat)

conf_plot = ConfusionMatrixDisplay(
    confusion_matrix = conf,
)

conf_plot.plot()

# HACER CROSS VALIDATION 

# GUARDAR MODELO
