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

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        dZ2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X_train, y_train, X_test, y_test, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            for i in range(X_train.shape[0]):
                # Forward e Backward pass para treinamento
                x = X_train[i]
                x = x.reshape(1, -1)  # Converter em matriz de uma amostra
                y = y_train[i]
                output_train = self.forward(x)
                self.backward(x, y, learning_rate)
                train_error = np.mean(np.abs(output_train - y))
                self.train_errors.append(train_error)

                # Forward pass para teste
                output_test = self.forward(X_test)
                test_error = np.mean(np.abs(output_test - y_test))
                self.test_errors.append(test_error)

    def predict(self, X):
        return np.round(self.forward(X))

#
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
# # Plotar curvas de erro durante o treinamento
# plt.plot(mlp.train_errors, label='Erro de Treinamento')
# plt.plot(mlp.test_errors, label='Erro de Teste')
# plt.xlabel('Número de Amostras')
# plt.ylabel('Erro')
# plt.legend()
# plt.show()
