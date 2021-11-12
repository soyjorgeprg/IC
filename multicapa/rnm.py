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

def resultado(y):
    max_value = max(y)
    max_index = list(y).index(max_value)
    return max_index


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

    def normalizar(self):
        salida = []
        for imagen in self.datos:
            img = []
            for pixel in range(784):
                img.append(imagen[pixel] / 255)
            salida.append(img)
        return salida

    def predecir(self, datos):
        tags_interna = []
        for img in datos:
            tags_interna.append(self.prediccion(img))
        return tags_interna

    def prediccion(self, imagen):
        salida = [imagen]
        for i in range(self.num_layers): salida.append([])
        for layer in range(self.num_layers):
            salida[layer + 1] = np.append(salida[layer + 1], self.layers[layer].predecir(salida[layer]))
        return salida

    def activacion(self):
        salida = []
        for i in range(self.num_layers): salida.append([])
        for layer in range(self.num_layers):
            salida[layer] = self.layers[layer].activacionSigmoide()
        return salida

    def error(self, label, y):
        err = []
        prediccion = resultado(y[-1])
        for i in range(self.layers[-1].num_neuronas):
            err.append(label - prediccion)
        return err

    def backpropagation(self, error, y, z):
        for i in range(len(error)):
            self.layers[-1].delta[i] = error[i] * derivada_sigmoide(z[-1][i])
        for layer in reversed(range(1, self.num_layers)):
            self.layers[layer].propagacion(y[layer], self.layers[layer - 1], z, error)

    def updateW(self, eta, input):
        for layer in range(self.num_layers):
            self.layers[layer].correccion(eta, input[layer])

    def entrenamiento(self, imagen, labels, numEpoch):
        # entrada = self.normalizar()
        for epoch in range(numEpoch):
            # for img, label in zip(entrada, labels):
            for img, label in zip(imagen, labels):
                z = self.prediccion(img)
                y = self.activacion()

                error_final = self.error(label, y)
                self.backpropagation(error_final, y, z)

                self.updateW(self.learning_rate, z)

        return [ layer.w for layer in self.layers]
