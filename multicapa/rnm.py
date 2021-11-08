from mlxtend.data import loadlocal_mnist
import numpy as np

from capa import Capa

def load_data():
    datos, etiquetas = loadlocal_mnist(
        images_path='../data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte',
        labels_path='../data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')
    return datos, etiquetas


def load_train():
    datos, etiquetas = loadlocal_mnist(
        images_path='../data/train-images-idx3-ubyte/train-images.idx3-ubyte',
        labels_path='../data/train-labels-idx1-ubyte/train-labels.idx1-ubyte')
    return datos, etiquetas


def sigmoide(x):
    return 1 / (1 + np.exp(-x))


def derivada_sigmoide(x):
    return sigmoide(x) * (1 - sigmoide(x))


class RedNeuronal:

    def __init__(self, layers_def):
        # self.w = np.zeros(shape=(10, 784))
        self.learning_rate = 0.02
        self.layers = [Capa(num_neuronas) for num_neuronas in layers_def]
        # self.layers = list(reversed(layers_def))
        self.num_layers = len(self.layers)

    def normalizar(self, imagenes):
        salida = []
        for imagen in imagenes:
            img = []
            for pixel in range(784):
                img.append(imagen[pixel] / 255)
            salida.append(img)
        return salida

    def prediccion(self, imagen):
        # salida = np.zeros(10*self.num_layers).reshape(self.num_layers, 10)
        salida = [imagen]
        for i in range(self.num_layers):
            salida.append([])
        for layer in range(self.num_layers + 1):
            for num in range(self.num_layers):
                aux = [sigmoide(np.dot(self.layers[num].w, salida[layer]))]
                salida[layer + 1] = np.append(salida[layer + 1], aux)
        return salida

    def activacion(self, x, perceptrones):
        salida = np.zeros(10)
        for i in range(0, perceptrones):
            salida[i] = 1.0 if (x[i] > 0) else 0.0
        return salida

    def correccion(self, y, x, label, perceptrones):
        for i in range(0, perceptrones):
            if y[i] == 0 and label == i:
                self.w[i] += x
            elif y[i] == 1 and label != i:
                self.w[i] -= x
        return self.w

    def entrenamiento(self, imagen, labels, lr, numEpoch):
        lr = self.learning_rate
        # entrada = self.normalizar(imagen)
        for epoch in range(numEpoch):
            # for img, label in zip(entrada, labels):
            for img, label in zip(imagen, labels):
                z = self.prediccion(img)
                y = self.activacion(z, 10)
                self.w = self.correccion(y, img, label, 10)
        return self.w

