from rnm import RedNeuronal
from mlxtend.data import loadlocal_mnist


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


train, etiq = load_train()

rn = RedNeuronal([10, 5, 5], train)

salida = rn.entrenamiento(train, etiq, 0.2, 5)

print(salida)

jq = RedNeuronal([10, 20, 20])
