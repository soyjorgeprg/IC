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

    def __init__(self, layers_def, datos):
        self.learning_rate = 0.02
        self.datos = datos
        self.layers = []
        for num_capa in range(len(layers_def)):
            if num_capa == 0:
                self.layers.append(Capa(layers_def[num_capa], self.datos.shape[1]))
            else:
                self.layers.append(Capa(layers_def[num_capa], layers_def[num_capa - 1]))
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
        salida = [imagen]
        for i in range(self.num_layers): salida.append([])
        for layer in range(self.num_layers):
            salida[layer + 1] = np.append(salida[layer + 1], self.layers[layer].prediccionSigmoide(salida[layer]))
        return salida

    def backpropagation(self, y, x, label):
        for layer in range(self.num_layers):
            self.layers[layer].correccion(y[layer], x, label)
            pass

    def entrenamiento(self, imagen, labels, lr, numEpoch):
        lr = self.learning_rate
        # entrada = self.normalizar(imagen)
        for epoch in range(numEpoch):
            # for img, label in zip(entrada, labels):
            for img, label in zip(imagen, labels):
                y = self.prediccion(img)
                self.backpropagation(y, img, label)
        return self.w

