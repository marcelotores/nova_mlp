import numpy as np
import matplotlib.pyplot as plt

import ut

# Gerando um conjunto de dados aleatório com 10 amostras e 24 atributos
np.random.seed(42)
dataset = np.random.rand(10, 24)

dataSet = ut.im_data(4)

### Divisão das amostras em 4 classes de 60 amostras cada
c1 = dataSet[:60, :]
c2 = dataSet[82:142, :]
c3 = dataSet[175:235, :]
c4 = dataSet[315:375, :]
dataset = np.concatenate((c1, c2, c3, c4), axis=0)

# Definindo o número total de atributos
num_attributes = dataset.shape[1]

# Criando todas as combinações possíveis de atributos par a par
attribute_combinations = [(i, j) for i in range(num_attributes) for j in range(i+1, num_attributes)]

# Plotando o gráfico de dispersão para cada par de atributos
for idx, (x_index, y_index) in enumerate(attribute_combinations):
    # Obtendo os valores dos atributos selecionados
    x = dataset[:, x_index]
    y = dataset[:, y_index]

    # Criando uma nova figura para cada plot
    plt.figure()

    # Criando um array 'c' com valores binários para as cores
    c = np.zeros_like(x)
    c[y_index] = 1

    # Plotando o gráfico de dispersão com duas cores, uma para cada atributo
    plt.scatter(x, y, c=c, cmap='coolwarm')

    # Adicionando rótulos aos eixos
    plt.xlabel(f'Atributo {x_index}')
    plt.ylabel(f'Atributo {y_index}')

    # Exibindo o gráfico de dispersão
    plt.show()
