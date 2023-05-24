import numpy as np
import matplotlib.pyplot as plt

import ut


def scatter_plot_classes(data, target, class_names, attribute_index1, attribute_index2):
    # Obtendo os valores dos atributos para os índices especificados
    x = data[:, attribute_index1]
    y = data[:, attribute_index2]

    # Criando uma lista de cores para as classes
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'yellow'] # Adicione mais cores se necessário

    # Plotando o gráfico de dispersão com cores distintas para cada classe
    for class_index in np.unique(target):
        plt.scatter(x[target == class_index], y[target == class_index], c=colors[class_index], label=class_names[class_index])

    # Adicionando rótulos aos eixos
    plt.xlabel(f'Atributo {attribute_index1}')
    plt.ylabel(f'Atributo {attribute_index2}')

    # Adicionando uma legenda
    plt.legend()

    # Exibindo o gráfico
    plt.show()


dataSet = ut.im_data(4)

### Divisão das amostras em 4 classes de 60 amostras cada
c1 = dataSet[:60, :]
c2 = dataSet[82:142, :]
c3 = dataSet[175:235, :]
c4 = dataSet[315:375, :]
print(dataSet)
exit()
dataset = np.concatenate((c1, c2, c3, c4), axis=0)
scatter_plot_classes(meu_dataset, meus_rótulos, class_names, 0, 1)
