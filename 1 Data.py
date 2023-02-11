"""
## Setting el dataset

Se usara la base de datos de `The Mist Database` disponible en la librería keras.
La base de datos se encuentra divida en 60_000 imagenes de _training_ y 10_000 de
_testing_. Para este proyecto se unira toda la base y se separara posteriormente
con `sklearn`.

Dado que el dataset original contiene imagenes de 28x28 pixeles, al transformarlos
a formato numerico se colocan de manera horizontal donde 28x28 = 784. Las columnas
representan los pixeles que han sido vectorizados. 

Al realizar el train_test_split, se reserva un 20% de la muestra para testear los
resultados. El training sample tendrá 56 mil observaciones.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X = np.append(X_train, X_test).reshape(70_000, 28, 28)
y = np.append(y_train, y_test)


# Propia separación
X_train, X_test, y_train, y_test\
    = train_test_split(X, y, test_size=0.2, random_state=19)


# A DataFrames para ser guardados
X_train_df = pd.DataFrame(X_train.reshape(56_000, 784)) 
y_train_df = pd.DataFrame(y_train.reshape(56_000,))
X_test_df  = pd.DataFrame(X_test.reshape(14_000, 784))
y_test_df  = pd.DataFrame(y_test.reshape(14_000,))

X_train_df.to_pickle("./data/X_train.pkl")
y_train_df.to_pickle("./data/y_train.pkl")
X_test_df.to_pickle("./data/X_test.pkl")
y_test_df.to_pickle("./data/y_test.pkl")



# Grafica 1 ========================================================
m, pix, pix = X_train.shape

fig, axs = plt.subplots(6, 6, figsize=(6,6))
plt.tight_layout()
np.random.seed(19) 

for i, ax in enumerate(axs.flat):
    random_index = np.random.randint(m)

    X_resized = X_train[random_index]
    ax.imshow(X_resized, cmap="gray")
    
    ax.set_title(int(y_train[random_index])) # Labels
    ax.set_axis_off()

plt.savefig("./figures/digits.png")



## Referencias:
# https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch13/mnist_keras_mlp.py
# https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
# https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6