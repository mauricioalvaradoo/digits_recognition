import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from functions import model_selected


# Data =====================================================================
X_train = pd.read_pickle("./data/X_train.pkl")
y_train = pd.read_pickle("./data/y_train.pkl")
X_test = pd.read_pickle("./data/X_test.pkl")
y_test = pd.read_pickle("./data/y_test.pkl")

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)


"""
# Estimación de modelo seleccionado ========================================
# La propuesta del modelo 2 terminó siendo la seleccionada.

Modelo base (2):
    * Layer0 = Insumos
    * Layer1 = 60 neuronas
    * Layer2 = 30 neuronas
    * Layer3 = 20 neuronas
    * Layer4 = 10 neuronas  

"""

models, results, fx = model_selected.fit(X_train, y_train, epochs=100, iters=50)


# Reestructurando los resultados
list_iters = []
list_loss = []

for i in results:
    list_iters.append(list(i.keys())[0])
    list_loss.append(list(i.values())[0])

# Filas: Iteraciones a diferentes seeds , Columnas: Epochs
df = pd.DataFrame(index = list_iters,  data = list_loss)
df.to_csv("./data/loss-selected-models.csv")

df.head(10)


# Función de pérdida de modelo seleccionado ================================
df_std_01 = df.quantile(q=0.01)
df_std_16 = df.quantile(q=0.16)
df_median = df.quantile(q=0.5)
df_std_84 = df.quantile(q=0.84)
df_std_99 = df.quantile(q=0.99)

x = [int(i) for i in np.linspace(1, 100, 100)]


plt.figure(figsize=(6,4))

plt.fill_between(x, df_std_01, df_std_99, color="red", alpha=0.2, label="98% confianza")
plt.fill_between(x, df_std_16, df_std_84, color="red", alpha=0.6, label="68% confianza")
plt.plot(x, df_median, color="black", label="mediana")
plt.xlim([0, 21])
plt.xticks(np.arange(0, 21, step=4)) # Limit epochs: 21

plt.xlabel("epochs")
plt.ylabel("magnitud")
plt.title("Funciones de pérdida por epochs")
plt.savefig("./figures/log-loss.pdf")
plt.legend(loc="best", fontsize=8)

plt.show()



# Gráfico label, predict


# Gráfico de observaciones con erroresr

