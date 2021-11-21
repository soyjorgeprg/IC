import numpy as np

def sigmoide(x):
    return 1 / (1 + np.exp(-x))


def derivada_sigmoide(x):
    return sigmoide(x) * (1 - sigmoide(x))

class Capa:

    def __init__(self, neuronas, tam_entrada):
        self._w = np.random.uniform(-1, 1, (neuronas, tam_entrada))
        self._num_neuronas = neuronas
        self._z = np.zeros(self._num_neuronas)
        self._y = np.zeros(self._num_neuronas)
        self._delta = np.random.uniform(0, 1, tam_entrada)
        self.error = np.random.uniform(0, 1, self._num_neuronas)

    @property
    def num_neuronas(self):
        return self._num_neuronas

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, d):
        self._delta = d

    @property
    def w(self):
        return self._w

    @property
    def z(self):
        return self._z

    @property
    def y(self):
        return self._y


    def predecir(self, imagen):
        for i in range(self.num_neuronas):
            self._z[i] = np.dot(imagen, self.w[i])
        return self._z

    def activacionSigmoide(self):
        for i in range(self.num_neuronas):
            self._y[i] = sigmoide(self._z[i])
        return self._y

    def prediccionSigmoide(self, imagen):
        for i in range(self.num_neuronas):
            self._y[i] = sigmoide(np.dot(imagen, self.w[i]))
        return self._y

    def propagacion(self, y, capa_sig, z, error):
        for i in range(self.num_neuronas):
            self.error[i] = np.dot(capa_sig.w[i], capa_sig.delta)
            self.delta[i] = self.error[i] * derivada_sigmoide(self.z[i])

    def correccion(self, eta, img):
        for i in range(self.num_neuronas):
            self.w[i] += eta * img * self.delta[i]