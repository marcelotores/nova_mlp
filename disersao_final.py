import numpy as np
import matplotlib.pyplot as plt

import ut
dataSet = ut.im_data(4)

##########3


##########3

#dataSet_3_classes = dataSet[:315, :]
dataSet_3_classes = dataSet
data = dataSet_3_classes[:, :24]

class_labels = dataSet_3_classes[:, 24].reshape(dataSet_3_classes.shape[0], 1)

# dataSet = ut.im_data(4)
# classe1 = dataSet[:82, :]
# classe2 = dataSet[315:, :]
# classe1_classe2 = np.concatenate((classe1, classe2), axis=0)
#
# data = classe1_classe2[:, :24]
# class_labels = classe1_classe2[:, 24].reshape(classe1_classe2.shape[0], 1)


# Definir cores para cada classe
colors = ['red', 'green', 'orange', 'blue']

# Plotar gráfico de dispersão
fig, ax = plt.subplots()
#classes_to_plot = [1, 2]
for i in range(0, 24):
    for class_label in range(1, 5):
        # Selecionar amostras da classe atual
        class_data = data[class_labels.flatten() == class_label]
        # Extrair atributos para os eixos x e y
        x = class_data[:, 0]  # Atributo 1
        y = class_data[:, i]  # Atributo 2

        # Plotar os pontos da classe atual
        ax.scatter(x, y, c=colors[class_label-1], label=f'Classe {class_label}')

ax.set_xlabel('Atributo 1')
ax.set_ylabel('Atributo 2')
ax.legend()
plt.show()

