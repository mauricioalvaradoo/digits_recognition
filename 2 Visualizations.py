"""""
Cada observacion dentro de la variable `X` representa un imagen.
En este caso, hay 56 mil imagenes. Las columnas representan los
pixeles que han sido vectorizados. Dado que el dataset original
contiene imagenes de 28x28 pixeles, al transformarlos a formato
numerico se colocan de manera horizontal donde 28x28 = 784.

"""""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Data
X_train = pd.read_pickle("./data/X_train.pkl")
y_train = pd.read_pickle("./data/y_train.pkl")
X_test = pd.read_pickle("./data/X_test.pkl")
y_test = pd.read_pickle("./data/y_test.pkl")

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)


# Grafica 1 ========================================================
m, n = X_train.shape
pixeles = 28

fig, axs = plt.subplots(6,6, figsize=(6,6))
fig.tight_layout(pad=0.1)
np.random.seed(19) 

for i, ax in enumerate(axs.flat):
    random_index = np.random.randint(m)

    X_resized = X_train[random_index].reshape((pixeles,pixeles))
    ax.imshow(X_resized, cmap='gray')
    
    ax.set_title(int(y_train[random_index])) # Labels
    ax.set_axis_off()

plt.savefig("./figures/digits.pdf")


