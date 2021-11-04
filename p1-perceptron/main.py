import numpy
from mlxtend.data import loadlocal_mnist
import numpy as np
import time

def load_train():
    datos, etiquetas = loadlocal_mnist(
        images_path='../data/train-images-idx3-ubyte/train-images.idx3-ubyte',
        labels_path='../data/train-labels-idx1-ubyte/train-labels.idx1-ubyte')
    return datos, etiquetas


def load_data():
    datos, etiquetas = loadlocal_mnist(
        images_path='../data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte',
        labels_path='../data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')
    return datos, etiquetas


def normalizar(imagenes):
    salida = []
    for imagen in imagenes:
        img = []
        for pixel in range(784):
            img.append(imagen[pixel] / 255)
        salida.append(img)
    return salida


def normalizar_salida(salida):
    suposicion = []
    for prediccion in salida:
        suposicion.append(1.0) if (prediccion > 0) else suposicion.append(0.0)
    return suposicion


def prediccion(imagen, pesos_, perceptrones):
    salida = np.zeros(10)
    for i in range(0, perceptrones):
        salida[i] = np.dot(imagen, pesos_[i])
    return salida


def predecir(imagenes, pesos_internos, perceptrones):
    tags_interna = []
    for img in imagenes:
        tags_interna.append(prediccion(img, pesos_internos, perceptrones))
    return tags_interna


def activacion(x, perceptrones):
    salida = np.zeros(10)
    for i in range(0, perceptrones):
        salida[i] = 1.0 if (x[i] > 0) else 0.0
    return salida


def correccion(y, x, pesos_gen, label, perceptrones):
    for i in range(0, perceptrones):
        if y[i] == 0 and label == i:
            pesos_gen[i] += x
        elif y[i] == 1 and label != i:
            pesos_gen[i] -= x
    return pesos_gen


def perceptron(numEpoch, imgs, labels):
    w = np.zeros(shape=(10, imgs.shape[1]))
    entrada = normalizar(imgs)
    for epoch in range(numEpoch):
        for img, label in zip(entrada, labels):
            z = prediccion(img, w, 10)
            y = activacion(z, 10)
            w = correccion(y, img, w, label, 10)
    return w


def acierto(tags_data, tags_final):
    correct = 0
    for (x, y) in zip(tags_data, tags_final):
        if y == x:
            correct += 1
    return correct


def traduccion(resultados):
    etiquetas = []
    for x in resultados:
        max_value = max(x)
        max_index = list(x).index(max_value)
        etiquetas.append(max_index)

    return etiquetas


# Cargamos los datos
train, etiq = load_train()
imgs, tags = load_data()

# Entrenamos la red
start = time.time()
pesos = perceptron(5, train, etiq)
done = time.time()
elapsed = done - start
print("Tiempo de entrenamiento: " + str(elapsed))

# Predecimos cada uno de los grupos
etiquetas_train = predecir(train, pesos, 10)
etiquetas_data = predecir(imgs, pesos, 10)

e_t = traduccion(etiquetas_train)
e_d = traduccion(etiquetas_data)

# Calculo de los aciertos
aciertos = acierto(e_t, etiq)
print(aciertos/len(etiq))
aciertos = acierto(e_d, tags)
print(aciertos/len(tags))

file = open("./etiquetas.txt", "a+")
for tag in e_d:
    file.write(str(tag))
file.close()



# IDEAS:
#   * https://towardsdatascience.com/perceptron-and-its-implementation-in-python-f87d6c7aa428
#   * https://www.llipe.com/2017/04/19/programando-un-clasificador-perceptron-en-python/
