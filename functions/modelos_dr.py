import numpy as np
import random
import math

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam



def model_selection_list():
    
    random_seed = int(random.choice(np.linspace(1, 10000, 10000)))
    tf.random.set_seed(random_seed)


    model_1 = Sequential(
        [
            Dense(40, activation = "relu"),
            Dense(20, activation = "relu"),
            Dense(10, activation = "linear")
        ],
        name="model_selection_1"
    )

    model_2 = Sequential(
        [
            Dense(60, activation = "relu"),
            Dense(30, activation = "relu"),
            Dense(20, activation = "relu"),
            Dense(10, activation = "linear")
        ],
        name="model_selection_1"
    )

    model_3 = Sequential(
        [
            Dense(20, activation = "relu"),
            Dense(20, activation = "relu"),
            Dense(15, activation = "relu"),
            Dense(15, activation = "relu"),
            Dense(10, activation = "linear")
        ],
        name="model_selection_3"
    )

    list_models = [model_1, model_2, model_3]
    
    return list_models




def fit_selected(X_train, y_train, epochs=50, iters=100):
    
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
                Input(shape = (784,)),
                Dense(units = 40, activation = "relu", name = "Layer1"),
                Dense(units = 20, activation = "relu", name = "Layer2"),
                Dense(units = 10,  activation = "linear", name = "Layer3")
            
            ], name = "Base"
        )

        model.compile(
            loss = SparseCategoricalCrossentropy(from_logits=True),
            optimizer = Adam(0.01),
        )

        results = model.fit(
            X_train, y_train,
            epochs = epochs,
            verbose = False
        )

        logits = model.predict(X_train)
        fx = tf.nn.softmax(logits)
        
        # Guardando
        store_model.append(model)
        loss = results.history["loss"]
        store_loss.append(loss)
        store_fx.append(fx)
    
        if i% math.ceil(iters/10) == 0:
            print(f"Iteración #{i:4} finalizada")


    return store_model, store_loss, store_fx