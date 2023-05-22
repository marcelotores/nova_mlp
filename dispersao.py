import numpy as np
import matplotlib.pyplot as plt

import ut

# # Gerar dados de exemplo
# np.random.seed(42)
#
# # Criar três classes
# num_samples = 100
# class_labels = np.random.randint(0, 3, num_samples)
#
# # Gerar dados aleatórios para cada atributo
# num_attributes = 24
# data = np.random.randn(num_samples, num_attributes)

##################3
# dataSet = ut.im_data(4)
# dataSet_3_classes = dataSet[:315, :]
# X = dataSet_3_classes[:, :24]
# y = dataSet_3_classes[:, 24].reshape(dataSet_3_classes.shape[0], 1)
#######
data = [[2.34148759e-03, 2.48435659e+00, 3.96172138e+08, 1.31843097e+04,
  5.33112720e-01, 1.00000000e+00, 5.17245129e+04, 2.26348078e+00,
  2.77132865e+00, 1.15404970e-03, 5.91052859e-01, 5.88400991e-01,
  1.00000000e+00, 1.00000000e+00, 5.75196927e-01, 1.12209717e+02,
  5.86893340e-03, 5.27318212e+04, 1.28738856e+07, 3.24454796e+09,
  1.17578597e+00, 3.89183851e-03, 1.31832108e+04, 5.78451997e-10],
 [1.76124638e-03, 3.57921097e+00, 5.47622634e+08, 1.57173902e+04,
  4.82912651e-01, 1.00000000e+00, 6.17348578e+04, 2.30358288e+00,
  2.88598704e+00, 1.01015859e-03, 6.55669947e-01, 5.59400298e-01,
  1.00000000e+00, 1.00000000e+00, 5.35587819e-01, 1.22730859e+02,
  4.13639881e-03, 6.28565989e+04, 1.66679767e+07, 4.54936877e+09,
  1.41083706e+00, 9.24710052e-03, 1.57142137e+04, 8.33375426e-10]]

class_labels = [[1.], [1.]]
#########
# novoy = []
# for i in y:
#     novoy.append(i[0])
# y = (np.rint(y)).astype(int)
# data = X
# class_labels = novoy
# print(data[:2, :])
# exit()
# print(class_labels)
# exit()
# print(y)
# exit()

################
# Definir cores para cada classe
colors = ['red', 'green', 'blue']

# Plotar gráfico de dispersão
fig, ax = plt.subplots()

for class_label in range(3):
    # Selecionar amostras da classe atual
    class_data = data[class_labels == class_label]

    # Extrair atributos para os eixos x e y
    x = class_data[:, 0]  # Atributo 1
    y = class_data[:, 1]  # Atributo 2

    # Plotar os pontos da classe atual
    ax.scatter(x, y, c=colors[class_label], label=f'Classe {class_label}')

ax.set_xlabel('Atributo 1')
ax.set_ylabel('Atributo 2')
ax.legend()
plt.show()