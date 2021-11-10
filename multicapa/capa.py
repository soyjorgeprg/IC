import numpy as np

def sigmoide(x):
    return 1 / (1 + np.exp(-x))


def derivada_sigmoide(x):
    return sigmoide(x) * (1 - sigmoide(x))

class Capa:

    def __init__(self, neuronas, tam_entrada):
        self._w = np.zeros(shape=(neuronas, tam_entrada))
        self._num_neuronas = neuronas
        self.salida = np.zeros(self._num_neuronas)

    @property
    def num_neuronas(self):
        return self._num_neuronas

    @property
    def w(self):
        return self._w


    def prediccionSigmoide(self, imagen):
        for i in range(self.num_neuronas):
            self.salida[i] = sigmoide(np.dot(imagen, self.w[i]))
        return self.salida

    def correccion(self, y, x, expected):
        for i in range(self.num_neuronas):
            if y[i] == 0 and expected == i:
                self.w[i] += x
            elif y[i] == 1 and expected != i:
                self.w[i] -= x
        return self.w