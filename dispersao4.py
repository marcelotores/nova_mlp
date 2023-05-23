import numpy as np
import matplotlib.pyplot as plt

import ut

# Gerando um conjunto de dados aleatório com 100 amostras e 2 atributos
np.random.seed(42)
dataset = np.random.rand(100, 2)

dataSet = ut.im_data(4)

### Divisão das amostras em 4 classes de 60 amostras cada
c1 = dataSet[:60, :]
c2 = dataSet[82:142, :]
c3 = dataSet[175:235, :]
c4 = dataSet[315:375, :]
dataset = np.concatenate((c1, c2, c3, c4), axis=0)

# Obtendo os valores dos atributos
x = dataset[:, 0]
y = dataset[:, 7]

# Criando um array de índices de cores para cada atributo
colors = np.zeros_like(x)
colors[y > x] = 1

# Plotando o gráfico de dispersão com cores distintas para cada atributo
plt.scatter(x, y, c=colors, cmap='coolwarm', s=50)

# Adicionando rótulos aos eixos
plt.xlabel('Atributo 1')
plt.ylabel('Atributo 2')

# Exibindo o gráfico
plt.show()
