import numpy as np


class Capa:

    def __init__(self, neuronas):
        self._w = np.zeros(shape=(10, neuronas))
        self._num_neuronas = neuronas

    @property
    def num_neuronas(self):
        return self._num_neuronas

    @property
    def w(self):
        return self._w
