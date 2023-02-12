import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback



def selection_list():
    
    random_seed = int(random.choice(np.linspace(1, 10_000, 10_000)))
    tf.random.set_seed(random_seed)


    model_1 = Sequential(
        [
            Conv2D(32, (3,3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3,3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(40, activation = "relu", name="Layer1"),
            Dense(20, activation = "relu", name="Layer2"),
            Dense(10, activation = "linear", name="Layer3")
        ],
        name="model_selection_1"
    )

    model_2 = Sequential(
        [
            Conv2D(32, (3,3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3,3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(60, activation = "relu", name="Layer1"),
            Dense(30, activation = "relu", name="Layer2"),
            Dense(20, activation = "relu", name="Layer3"),
            Dense(10, activation = "linear", name="Layer4")
        ],
        name="model_selection_2"
    )

    model_3 = Sequential(
        [
            Conv2D(32, (3,3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3,3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(20, activation = "relu", name="Layer1"),
            Dense(20, activation = "relu", name="Layer2"),
            Dense(15, activation = "relu", name="Layer3"),
            Dense(15, activation = "relu", name="Layer4"),
            Dense(10, activation = "linear", name="Layer5")
        ],
        name="model_selection_3"
    )

    list_models = [model_1, model_2, model_3]
    
    return list_models



def model_selected(X_train, y_train, epochs=100, iters=50, seed=None):
    
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
        Historia de los modelos
    store_iters: list
        Historia de las seeds que generan cada cadena
    store_loss: list
        Historia de las funciones de pérdida por cadena
    store_fx: list
        Historia de las estimaciones por cadena

    Arquitectura neuronal
    ---------------------
    * Convolucional = 32 capas
    * Pooling = 2x2
    * Convolucional = 64 capas
    * Pooling = 2x2
    * Layer1 = 60 neuronas
    * Layer2 = 30 neuronas
    * Layer3 = 20 neuronas
    * Layer4 = 10 neuronas

    """
    
    # Callbacks -> Frenar estimación cuando se alcance MSE (0.01) o ajuste (99.5%)
    class myCallback(Callback):
        def on_epoch_end(self, epoch, logs={}):
            if ((logs.get("loss")<0.01) or (logs.get("accuracy")>0.995)):
                self.model.stop_training = True
    callbacks = myCallback()


    store_model = []
    list_loss = []
    store_fx = []
    list_acc = []

    for i in range(1, iters+1):
        
        
        if seed is None:
            random_seed = int(random.choice(np.linspace(1, 10_000, 10_000)))
            tf.random.set_seed(random_seed)
        else:
            random_seed = seed
            tf.random.set_seed(random_seed)
            
            
        model = Sequential(
            [
                Conv2D(32, (3,3), activation="relu", input_shape=(28, 28, 1)),
                MaxPooling2D(2, 2),
                Conv2D(64, (3,3), activation="relu", input_shape=(28, 28, 1)),
                MaxPooling2D(2, 2),
                Flatten(),
                Dense(60, activation = "relu"),
                Dense(30, activation = "relu"),
                Dense(20, activation = "relu"),
                Dense(10, activation = "linear")
            ],
            name = "base"
        )

        model.compile(
            loss = SparseCategoricalCrossentropy(from_logits=True),
            optimizer = Adam(0.001),
            metrics = ["mean_squared_error", "accuracy"]
        )

        results = model.fit(
            X_train, y_train,
            epochs = epochs,
            verbose = False,
            callbacks = [callbacks]
        )

        logits = model.predict(X_train)
        fx_train = tf.nn.softmax(logits)
        
        # Guardando
        store_model.append({random_seed: model})
        loss = results.history["loss"]
        list_loss.append({random_seed: loss})
        store_fx.append({random_seed: fx_train})
        list_acc.append({random_seed: results.history["accuracy"][-1]})
    
        if i% math.ceil(iters/10) == 0:
            print(f"Iteración #{i:3} finalizada")


    store_iters = []
    store_loss = []
    store_acc = []

    for j in list_loss:
        store_iters.append(list(j.keys())[0])
        store_loss.append(list(j.values())[0])
    
    for j in list_acc:
        store_acc.append(list(j.values())[0])
    

    return store_model, store_iters, store_loss, store_acc, store_fx




def to_figure(
    data, col_width=2.0, row_height=0.5, font_size=10,
    header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
    bbox=[0, 0, 1, 1], header_columns=0, ax=None, **kwargs):
    
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    
    return ax.get_figure(), ax