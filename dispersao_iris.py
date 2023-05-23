import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import ut

# Carregando o conjunto de dados iris
iris = load_iris()
data = iris.data

target = iris.target
class_names = iris.target_names

#########################
dataSet = ut.im_data(4)

### Divisão das amostras em 4 classes de 60 amostras cada
c1 = dataSet[:60, :]
c2 = dataSet[82:142, :]
c3 = dataSet[175:235, :]
#c4 = dataSet[315:375, :]
dataset = np.concatenate((c1, c2, c3), axis=0)
#dataset /= np.max(dataset, axis=0)

data = dataset[:, :24]
target = dataset[:, 24]
target = target.astype(int)
class_names = ['classe1', 'classe2', 'classe3']


#########################
# Obtendo os valores dos atributos para os dois primeiros atributos
x = data[:, 0]
y = data[:, 1]
print(x)
exit()
# Criando uma lista de cores para as classes
colors = ['blue', 'red', 'green']

# Plotando o gráfico de dispersão com cores distintas para cada classe
for class_index in np.unique(target):
    plt.scatter(x[target == class_index], y[target == class_index], c=colors[class_index], label=class_names[class_index])

# Adicionando rótulos aos eixos
plt.xlabel('Atributo 1')
plt.ylabel('Atributo 2')

# Adicionando uma legenda
plt.legend()

# Exibindo o gráfico
plt.show()
