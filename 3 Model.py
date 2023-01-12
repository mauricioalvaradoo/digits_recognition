import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
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

models, iters, loss, fx = model_selected.fit(X_train, y_train, epochs=100, iters=100)


# Filas: Iteraciones a diferentes seeds , Columnas: Epochs
df = pd.DataFrame(index = iters,  data = loss)
df.to_csv("./results/model_selected_loss.csv")
df.head(10)

# df = pd.read_csv("./results/model_selected_loss.csv", index_col="Unnamed: 0")


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
plt.legend(loc="best", fontsize=8)

plt.savefig("./figures/selected-loss.png")
plt.show()


# Recuperamos modelo =======================================================
# Usaré la combinación de uno de los seed ya estimados: "150"
models, iters, loss, fx = model_selected.fit(X_train, y_train, epochs=100, iters=1, seed=150)
model = models[0][150]

yhat_train = []
for j in range(len(fx[0][150])):
    yhat_train.append(np.argmax(fx[0][150][j]))

yhat_train = np.array([np.asarray(yhat_train)]).T
error_train = (yhat_train != y_train) # Error del training set!
mse_train = np.mean(error_train) # MSE training: % de errores/total


# Testing
logits = model.predict(X_test)
fx_test = tf.nn.softmax(logits)

yhat_test = []
for j in range(len(fx_test)):
    yhat_test.append(np.argmax(fx_test[j]))

yhat_test = np.array([np.asarray(yhat_test)]).T
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
m, n = X_test.shape
pixeles = 28

fig, axes = plt.subplots(6,6, figsize=(6,6))
fig.tight_layout(pad=0.13,rect=[0, 0.03, 1, 0.85])
np.random.seed(19) 

for i, ax in enumerate(axes.flat):
    random_index = np.random.randint(m)

    X_resized = X_test[random_index].reshape((pixeles,pixeles))
    ax.imshow(X_resized, cmap="gray")

    ax.set_title(f"{y_test[random_index, 0]},{yhat_test[random_index, 0]}",fontsize=10)
    ax.set_axis_off()
fig.suptitle("Hits\nLabel, yhat", fontsize=14)
plt.savefig("./figures/model_selection_digits_hits.png")



# Gráfico label vs predict (errores) =======================================
pixeles = 28

idxs = np.where(yhat_test[:,0] != y_test[:,0])[0]
if len(idxs) == 0:
    print("No se encontraron errores!")
else:
    cnt = min(7, len(idxs))
    fig, ax = plt.subplots(1,cnt, figsize=(5,2.2))

for i in range(cnt):
    j = idxs[i]

    X_reshaped = X_test[j].reshape((pixeles,pixeles))
    ax[i].imshow(X_reshaped, cmap='gray')

    ax[i].set_title(f"{y_test[j,0]},{yhat_test[j,0]}",fontsize=10)
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
