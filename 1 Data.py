"""""
## Setting el dataset

Se usara la base de datos de `The Mist Database`: http://yann.lecun.com/exdb/mnist/.
La base de datos se encuentra divida en 60_000 imagenes de _training_ y 10_000 de
_testing_. Para este proyecto se unira toda la base y se separara posteriormente
con `sklearn`.

Dado que el dataset original contiene imagenes de 28x28 pixeles, al transformarlos
a formato numerico se colocan de manera horizontal donde 28x28 = 784. En este caso,
hay 56 mil imagenes. Las columnas representan los pixeles que han sido vectorizados. 
"""""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from functions import load_data 
from sklearn.model_selection import train_test_split


X_train, y_train, X_test, y_test = load_data.get_images("./data/")


X = np.append(X_train, X_test).reshape(70_000, 784)
y = np.append(y_train, y_test)

X_df = pd.DataFrame(X)
y_df = pd.DataFrame(y)


# Propia separacion
X_train, X_test, y_train, y_test\
    = train_test_split(X_df, y_df, test_size=0.3, random_state=19)


# Reset index
X_train = X_train.reset_index().drop("index", axis=1)
y_train = y_train.reset_index().drop("index", axis=1)
X_test = X_test.reset_index().drop("index", axis=1)
y_test = y_test.reset_index().drop("index", axis=1)


X_train.to_pickle("./data/X_train.pkl")
y_train.to_pickle("./data/y_train.pkl")
X_test.to_pickle("./data/X_test.pkl")
y_test.to_pickle("./data/y_test.pkl")



# Grafica 1 ========================================================
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

m, n = X_train.shape
pixeles = 28

fig, axs = plt.subplots(6, 6, figsize=(6,6))
plt.tight_layout()
np.random.seed(19) 

for i, ax in enumerate(axs.flat):
    random_index = np.random.randint(m)

    X_resized = X_train[random_index].reshape((pixeles,pixeles))
    ax.imshow(X_resized, cmap="gray")
    
    ax.set_title(int(y_train[random_index])) # Labels
    ax.set_axis_off()

plt.savefig("./figures/digits.pdf")



## Referencias:
# https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch13/mnist_keras_mlp.py
# https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
# https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6