import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
        self.W1 = np.random.normal(0, 1, (self.input_size, self.hidden_size)).astype(np.float128)
        self.b1 = np.zeros((1, self.hidden_size), dtype=np.float128)
        self.W2 = np.random.normal(0, 1, (self.hidden_size, self.output_size)).astype(np.float128)
        self.b2 = np.zeros((1, self.output_size), dtype=np.float128)
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
            # Forward e Backward pass para treinamento
            output_train = self.forward(X_train)
            self.backward(X_train, y_train, learning_rate)
            train_error = np.mean(np.abs(output_train - y_train))
            self.train_errors.append(train_error)

            # Forward pass para teste
            output_test = self.forward(X_test)
            test_error = np.mean(np.abs(output_test - y_test))
            self.test_errors.append(test_error)

    def predict(self, X):
        return np.round(self.forward(X))


# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Pré-processamento dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

# Converter os arrays para np.float128
X = X.astype(np.float128)
y = y.astype(np.float128)


# Dividir o conjunto de dados
