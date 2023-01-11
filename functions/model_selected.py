import numpy as np
import random
import math

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam



def fit(X_train, y_train, epochs=50, iters=100):
    
    """ Entrenando modelo base
    
    Parámetros
    -----------
    X_train: np.array
        Set de explicativas de entrenamiento
    y_train: np.array
        Set de target de entramiento
    epochs: int
        Cantidad de epochs en la estimación
    iter: int
        Número de iteraciones

    Retorno
    ----------
    store_model: list
        Modelos
    store_loss: list
        Historia de las funciones de pérdida por cadena
    store_fx: list
        Historia de las estimaciones por cadena

    Arquitectura neuronal
    ---------------------
    * Layer0 = Insumos
    * Layer1 = 40 neuronas
    * Layer2 = 20 neuronas
    * Layer3 = 10 neuronas


    """
    
    store_model = []
    store_loss = []
    store_fx = []

    for i in range(1, iters+1):
        
        random_seed = int(random.choice(np.linspace(1, 10000, 10000)))
        tf.random.set_seed(random_seed)

        model = Sequential(
            [
                Input(shape=(784,)),
                Dense(60, activation = "relu"),
                Dense(30, activation = "relu"),
                Dense(20, activation = "relu"),
                Dense(10, activation = "linear")
            ],
        name = "Base"
    )

        model.compile(
            loss = SparseCategoricalCrossentropy(from_logits=True),
            optimizer = Adam(0.001),
        )

        results = model.fit(
            X_train, y_train,
            epochs = epochs,
            verbose = False
        )

        logits = model.predict(X_train)
        fx = tf.nn.softmax(logits)
        
        # Guardando
        store_model.append({random_seed: model})
        loss = results.history["loss"]
        store_loss.append({random_seed: loss})
        store_fx.append({random_seed: fx})
    
        if i% math.ceil(iters/10) == 0:
            print(f"Iteración #{i:3} finalizada")


    return store_model, store_loss, store_fx