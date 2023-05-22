import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.random.uniform(-0.5, 0.5, size=(self.input_size, self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.uniform(-0.5, 0.5, size=(self.hidden_size, self.output_size))
        self.b2 = np.zeros((1, self.output_size))
        self.train_errors = []
        self.test_errors = []
        self.peso1 = []
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        for i in range(m):
            x = X[i]
            x = x.reshape(1, -1)  # Converter em matriz de uma amostra
            y_i = y[i]

            # Forward pass
            a2 = self.forward(x)

            # Backward pass
            dZ2 = a2 - y_i
            dW2 = np.dot(self.a1.T, dZ2)
            db2 = dZ2
            dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.a1)
            dW1 = np.dot(x.T, dZ1)
            db1 = dZ1

            # Atualizar pesos
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.peso1.append(self.W1)


    def train(self, X_train, y_train, X_test, y_test, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            # Forward pass para treinamento
            output_train = self.forward(X_train)
            train_error = np.mean(np.abs(output_train - y_train))
            self.train_errors.append(train_error)

            # Backward pass para treinamento
            self.backward(X_train, y_train, learning_rate)

            # Forward pass para teste
            output_test = self.forward(X_test)
            test_error = np.mean(np.abs(output_test - y_test))
            self.test_errors.append(test_error)


    def predict(self, X):
        return np.round(self.forward(X))


# # Carregar o conjunto de dados Iris
# data = load_iris()
# X = data.data
# y = data.target
#
# # Pré-processamento dos dados
# enc = OneHotEncoder(sparse=False)
# y = enc.fit_transform(y.reshape(-1, 1))
#
# # Dividir o conjunto de dados em treino e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Criar e treinar o modelo MLP
# input_size = X.shape[1]
# hidden_size = 16
# output_size = y.shape[1]
#
# mlp = MLP(input_size, hidden_size, output_size)
# learning_rate = 0.1
# num_epochs = 100
#
# mlp.train(X_train, y_train, X_test, y_test, learning_rate, num_epochs)
#
# # Fazer a predição
# y_pred_train = mlp.predict(X_train)
# y_pred_test = mlp.predict(X_test)
#
# # Calcular a acurácia
# accuracy_train = np.mean(y_pred_train == y_train)
# accuracy_test = np.mean(y_pred_test == y_test)
#
# print(f"Acurácia (Treinamento): {accuracy_train:.2%}")
# print(f"Acurácia (Teste): {accuracy_test:.2%}")
#
# # Plotar curva de erro por época
# epochs = range(1, num_epochs + 1)
# plt.plot(epochs, mlp.train_errors, label='Erro de Treinamento')
# plt.plot(epochs, mlp.test_errors, label='Erro de Teste')
# plt.xlabel('Época')
# plt.ylabel('Erro')
# plt.legend()
# plt.show()
