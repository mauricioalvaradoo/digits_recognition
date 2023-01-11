import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential, Input



def selection_list():
    
    random_seed = int(random.choice(np.linspace(1, 10000, 10000)))
    tf.random.set_seed(random_seed)


    model_1 = Sequential(
        [
            Input(shape=(784,)),
            Dense(40, activation = "relu", name="Layer1"),
            Dense(20, activation = "relu", name="Layer2"),
            Dense(10, activation = "linear", name="Layer3")
        ],
        name="model_selection_1"
    )

    model_2 = Sequential(
        [
            Input(shape=(784,)),
            Dense(60, activation = "relu", name="Layer1"),
            Dense(30, activation = "relu", name="Layer2"),
            Dense(20, activation = "relu", name="Layer3"),
            Dense(10, activation = "linear", name="Layer4")
        ],
        name="model_selection_2"
    )

    model_3 = Sequential(
        [
            Input(shape=(784,)),
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



