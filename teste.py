import numpy as np
from matplotlib import pyplot as plt
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

### Divisão das amosras em 2 classes (c1, c4)
classes_1_4 = np.concatenate((c1, c4), axis=0)
#X = classes_1_4[:, :24]
#y = classes_1_4[:, 24].reshape(classes_1_4.shape[0], 1)

### Divisão das amostras em 2 classes (c1, c3)

classes_1_3 = np.concatenate((c1, c4), axis=0)
X = classes_1_3[:, :24]
y = classes_1_3[:, 24].reshape(classes_1_3.shape[0], 1)

# Dividindo os rótulos em 4 atributos.
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

# Divindo os dados em testes e treinamento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Normalizando os dados
X_train /= np.max(X_train, axis=0)
X_test /= np.max(X_test, axis=0)

input_size = X_train.shape[1]
hidden_size = 50
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

mlp.train(X_train, y_train, X_test, y_test, learning_rate=0.01, num_epochs=5000)

# Fazer previsões no conjunto de teste
y_pred = mlp.predict(X_test)

# Calcular a acurácia
accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
print(f"Acurácia: {accuracy}")

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