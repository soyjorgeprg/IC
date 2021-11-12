from rnm import RedNeuronal
from mlxtend.data import loadlocal_mnist
import time


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

def traduccion(resultados):
    etiquetas = []
    for x in resultados:
        max_value = max(x)
        max_index = list(x).index(max_value)
        etiquetas.append(max_index)

    return etiquetas

def acierto(tags_data, tags_final):
    correct = 0
    for (x, y) in zip(tags_data, tags_final):
        if y == x:
            correct += 1
    return correct


train, etiq = load_train()

rn = RedNeuronal([10, 5, 5], train)

rn.entrenamiento(train, etiq, 5)

# Cargamos los datos
train, etiq = load_train()
imgs, tags = load_data()

# Creamos la red neuronal
rn = RedNeuronal([10, 5, 5], train)

# Entrenamos la red
start = time.time()
pesos = rn.entrenamiento(train, etiq, 5)
done = time.time()
elapsed = done - start
print("Tiempo de entrenamiento: " + str(elapsed))

# Predecimos cada uno de los grupos
etiquetas_train = rn.predecir(train)
etiquetas_data = rn.predecir(imgs)

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
