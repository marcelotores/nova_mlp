import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import ut
from mlp_amostra_3 import MLP
#from corri import MLP

dataSet = ut.im_data(4)

### Divisão das amostras em 4 classes de 60 amostras cada
c1 = dataSet[:60, :]
c2 = dataSet[82:142, :]
c3 = dataSet[175:235, :]
c4 = dataSet[315:375, :]
classes_4 = np.concatenate((c1, c2, c3, c4), axis=0)
#X = classes_4[:, :24]
#y = classes_4[:, 24].reshape(classes_4.shape[0], 1)

################# C1 C2
c1_c2 = np.concatenate((c1, c2), axis=0)
X = c1_c2[:, :24]
y = c1_c2[:, 24].reshape(c1_c2.shape[0], 1)

### Divisão das amosras em 2 classes (c1, c4)
classes_1_4 = np.concatenate((c1, c4), axis=0)
# X = classes_1_4[:, :24]
# y = classes_1_4[:, 24].reshape(classes_1_4.shape[0], 1)

### Divisão das amostras em 2 classes (c1, c3)

classes_1_3 = np.concatenate((c1, c4), axis=0)
# X = classes_1_3[:, :24]
# y = classes_1_3[:, 24].reshape(classes_1_3.shape[0], 1)

############ Classe C2 C3

c2_c3 = np.concatenate((c2, c3), axis=0)
# X = c2_c3[:, :24]
# y = c2_c3[:, 24].reshape(c2_c3.shape[0], 1)

############ Classe C3 C4
c3_c4 = np.concatenate((c2, c3), axis=0)
# X = c3_c4[:, :24]
# y = c3_c4[:, 24].reshape(c3_c4.shape[0], 1)

# Dividindo os rótulos em 4 atributos.
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

# Divindo os dados em testes e treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Normalizando os dados
X_train /= np.max(X_train, axis=0)
X_test /= np.max(X_test, axis=0)

input_size = X_train.shape[1]
hidden_size = 10
# 3
# 0.1
# teste 30%
output_size = y_train.shape[1]

mlp = MLP(input_size, hidden_size, output_size)

initial_learning_rate = 0.1
decay_rate = 0.01
num_epochs = 1000

#for epoch in range(num_epochs):
#    learning_rate = initial_learning_rate * np.exp(-decay_rate * epoch)
#    mlp.train(X_train, y_train, X_test, y_test, learning_rate, 1)

mlp.train(X_train, y_train, X_test, y_test, learning_rate=0.01, num_epochs=10000)

# Fazer previsões no conjunto de teste
y_pred = mlp.predict(X_test)


##################
# Fazer previsões no conjunto de teste
y_pred = mlp.predict(X_test)

# Calcular a acurácia
accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
print(f"Acurácia: {accuracy}")

# Converter as previsões de volta para as classes originais
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calcular a matriz de confusão
confusion_mat = confusion_matrix(y_test_classes, y_pred_classes)

# Exibir a matriz de confusão
print(confusion_mat)

confusion_mat = confusion_matrix(y_test_classes, y_pred_classes)

# Definir rótulos das classes
class_labels = ['Classe 1', 'Classe 2']

# Exibir a matriz de confusão
fig, ax = plt.subplots()
im = ax.imshow(confusion_mat, cmap='Blues')

# Adicionar barra de cores
cbar = ax.figure.colorbar(im, ax=ax)

# Configurar os ticks e rótulos do eixo x e y
ax.set_xticks(np.arange(len(class_labels)))
ax.set_yticks(np.arange(len(class_labels)))
ax.set_xticklabels(class_labels)
ax.set_yticklabels(class_labels)

# Rotacionar os rótulos do eixo x
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop para exibir os valores na matriz de confusão
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        text = ax.text(j, i, confusion_mat[i, j], ha="center", va="center", color="black")

# Configurar título e rótulos dos eixos
ax.set_title("Matriz de Confusão")
ax.set_xlabel("Valores Preditos")
ax.set_ylabel("Valores Reais")

# Exibir a figura
plt.show()
##################



# Plotar o gráfico de erro durante o treinamento e teste
#plt.subplot(1, 2, 1)
plt.plot(range(len(mlp.train_errors)), mlp.train_errors, label='Treinamento')
plt.plot(range(len(mlp.test_errors)), mlp.test_errors, label='Teste')
plt.xlabel('Época')
plt.ylabel('Erro')
plt.title('Gráfico de Erro durante o Treinamento')
plt.legend()

# Plotar o gráfico de acurácia durante o treinamento e teste
#plt.subplot(1, 2, 2)
#plt.plot(range(len(mlp.train_accuracies)), mlp.train_accuracies, label='Treinamento')
#plt.plot(range(len(mlp.test_accuracies)), mlp.test_accuracies, label='Teste')
#plt.xlabel('Época')
#plt.ylabel('Acurácia')
#plt.title('Gráfico de Acurácia durante o Treinamento e Teste')
#plt.legend()

#plt.tight_layout()
plt.show()