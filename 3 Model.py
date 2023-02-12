import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import utils


# Data =====================================================================
X_train = pd.read_pickle("./data/X_train.pkl")
y_train = pd.read_pickle("./data/y_train.pkl")
X_test = pd.read_pickle("./data/X_test.pkl")
y_test = pd.read_pickle("./data/y_test.pkl")

X_train = np.asarray(X_train).reshape(56_000, 28, 28)
y_train = np.asarray(y_train).reshape(56_000,)
X_test = np.asarray(X_test).reshape(14_000, 28, 28)
y_test = np.asarray(y_test).reshape(14_000,)


"""
# Estimación de modelo seleccionado ========================================
# La propuesta del modelo 2 terminó siendo la seleccionada.

Modelo base (2):
    * Convolucional = 32 capas
    * Pooling = 2x2
    * Convolucional = 64 capas
    * Pooling = 2x2
    * Layer1 = 60 neuronas
    * Layer2 = 30 neuronas
    * Layer3 = 20 neuronas
    * Layer4 = 10 neuronas

"""

models, iters, loss, acc, fx = utils.model_selected(
    X_train, y_train, epochs=100, iters=50
)


# Filas: Iteraciones a diferentes seeds , Columnas: Epochs
df = pd.DataFrame(index = iters,  data = loss)
df.to_csv("./results/model_selected_loss.csv")
df.head(10)

# df = pd.read_csv("./results/model_selected_loss.csv", index_col="Unnamed: 0")


# Función de pérdida de modelo seleccionado ================================
df_std_16 = df.quantile(q=0.16)
df_median = df.quantile(q=0.5)
df_std_84 = df.quantile(q=0.84)

x = [int(i) for i in np.linspace(0, 14, 15)]


plt.figure(figsize=(6,4))

plt.fill_between(x, df_std_16, df_std_84, color="red", alpha=0.5, label="68% confianza")
plt.plot(x, df_median, color="black", label="mediana")

plt.xlabel("epochs")
plt.ylabel("magnitud")
plt.title("Funciones de pérdida por epochs")
plt.legend(loc="best", fontsize=8)

plt.savefig("./figures/selected-loss.png")
plt.show()



# ============================================================================
# Recuperamos uno de los modelo ==============================================
# Usaré la combinación de uno de los seed ya estimados: "1000"
models, iters, loss, acc, fx = utils.model_selected(X_train, y_train, epochs=100, iters=1, seed=1000)
model = models[0][1000]

yhat_train = []
for j in range(len(fx[0][1000])):
    yhat_train.append(np.argmax(fx[0][1000][j]))

yhat_train = np.array(yhat_train)
error_train = (yhat_train != y_train) # Error del training set!
mse_train = np.mean(error_train) # MSE training: % de errores/total


# Testing
logits = model.predict(X_test)
fx_test = tf.nn.softmax(logits)

yhat_test = []
for j in range(len(fx_test)):
    yhat_test.append(np.argmax(fx_test[j]))

yhat_test = np.array(yhat_test)
error_test = (yhat_test != y_test) # Error del testing set!
mse_test = np.mean(error_test) # MSE testing: % de errores/total


# Tabla con aciertos y desaciertos en set training =========================
reporte_train = classification_report(y_train, yhat_train)
print(reporte_train) # Ajuste excelente!


conf_train = confusion_matrix(y_train, yhat_train)
plt.figure(figsize=(16,16))
fig = ConfusionMatrixDisplay(confusion_matrix = conf_train)
fig.plot(colorbar = False, cmap=plt.cm.Blues)
plt.title("Matriz de confusión")
plt.xlabel("Predicciones")
plt.ylabel("Etiquetas")
plt.savefig("./figures/confmatrix_train.png", bbox_inches="tight")


# Tabla con aciertos y desaciertos en set testing ==========================
reporte_test = classification_report(y_test, yhat_test)
print(reporte_test) # Ajuste excelente!


conf_test = confusion_matrix(y_test, yhat_test)
plt.figure(figsize=(16,16))
fig = ConfusionMatrixDisplay(confusion_matrix = conf_test)
fig.plot(colorbar = False, cmap=plt.cm.Blues)
plt.title("Matriz de confusión")
plt.xlabel("Predicciones")
plt.ylabel("Etiquetas")
plt.savefig("./figures/confmatrix_testing.png", bbox_inches="tight")



# Gráfico label vs predict (aciertos) ======================================
m, n, n = X_test.shape

fig, axes = plt.subplots(6,6, figsize=(6,6))
fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.85])
np.random.seed(19) 

for i, ax in enumerate(axes.flat):
    random_index = np.random.randint(m)

    X_resized = X_test[random_index]
    ax.imshow(X_resized, cmap="gray")

    ax.set_title(f"{y_test[random_index]},{yhat_test[random_index]}",fontsize=10)
    ax.set_axis_off()
fig.suptitle("Hits\nLabel, yhat", fontsize=14)
plt.savefig("./figures/model_selection_digits_hits.png")



# Gráfico label vs predict (errores) =======================================
idxs = np.where(yhat_test[:] != y_test[:])[0]
if len(idxs) == 0:
    print("No se encontraron errores!")
else:
    cnt = min(7, len(idxs))
    fig, ax = plt.subplots(1,cnt, figsize=(5,2.2))

for i in range(cnt):
    j = idxs[i]

    X_reshaped = X_test[j]
    ax[i].imshow(X_reshaped, cmap='gray')

    ax[i].set_title(f"{y_test[j]},{yhat_test[j]}",fontsize=10)
    ax[i].set_axis_off()
    fig.suptitle("Errors\nLabel, yhat", fontsize=12)

fig.tight_layout()    
plt.subplots_adjust(top=1, hspace=0.5, wspace=0.5)
plt.savefig("./figures/model_selection_digits_errors.png", bbox_inches="tight")



# Guardado modelo ==========================================================
model.save("./results/model.h5")

# !pip install tensorflowjs
# !mkdir results/tfjs_target
# !tensorflowjs_converter --input_format keras results/model.h5 results/tfjs_target
