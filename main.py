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
    def _init_(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
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
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
X_train /= np.max(X_train, axis=0)
X_test /= np.max(X_train, axis=0)

# Definir a arquitetura da MLP
input_size = X_train.shape[1]
hidden_size = 10
output_size = y_train.shape[1]

# Criar e treinar a MLP

mlp = MLP(input_size, hidden_size, output_size)
mlp.train(X_train, y_train, X_test, y_test, learning_rate=0.1, num_epochs=10000)

# Fazer previsões no conjunto de teste
y_pred = mlp.predict(X_test)

# Calcular a acurácia
accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
print(f"Acurácia: {accuracy}")

# Plotar o gráfico de erro durante o treinamento e teste
plt.plot(range(len(mlp.train_errors)), mlp.train_errors, label='Treinamento')
plt.plot(range(len(mlp.test_errors)), mlp.test_errors, label='Teste')
plt.xlabel('Época')
plt.ylabel('Erro')
plt.title('Gráfico de Erro durante o Treinamento e Teste')
plt.legend()
plt.show()
