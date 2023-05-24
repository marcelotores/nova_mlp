import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Carregando o conjunto de dados iris
iris = load_iris()
data = iris.data
target = iris.target
class_names = iris.target_names

# Obtendo os valores dos atributos para os dois primeiros atributos
x = data[:, 0]
y = data[:, 1]

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
