import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Carregando o conjunto de dados Iris
iris = load_iris()
X = iris.data
y = iris.target

# Codificando as classes usando One-Hot Encoding
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Função de ativação sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da função de ativação sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Classe MLP
class MLP:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # Inicializando os pesos aleatoriamente
        self.weights_hidden = np.random.rand(self.num_inputs, self.num_hidden)
        self.weights_output = np.random.rand(self.num_hidden, self.num_outputs)

        # Inicializando os biases aleatoriamente
        self.biases_hidden = np.random.rand(1, self.num_hidden)
        self.biases_output = np.random.rand(1, self.num_outputs)

    def forward(self, X):
        # Calculando a saída da camada oculta
        self.hidden_output = sigmoid(np.dot(X, self.weights_hidden) + self.biases_hidden)

        # Calculando a saída da camada de saída
        self.output = sigmoid(np.dot(self.hidden_output, self.weights_output) + self.biases_output)

    def backward(self, X, y, learning_rate):
        # Calculando o erro na camada de saída
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        # Calculando o erro na camada oculta
        hidden_error = np.dot(output_delta, self.weights_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Atualizando os pesos e os biases
        self.weights_output += learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.biases_output += learning_rate * np.sum(output_delta, axis=0)

        self.weights_hidden += learning_rate * np.dot(X.T, hidden_delta)
        self.biases_hidden += learning_rate * np.sum(hidden_delta, axis=0)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

    def predict(self, X):
        self.forward(X)
        return np.argmax(self.output, axis=1)

# Definindo os parâmetros da MLP
num_inputs = X_train.shape[1]
num_hidden = 5
num_outputs = y_train.shape[1]

# Criando uma instância da MLP
mlp = MLP(num_inputs, num_hidden, num_outputs)

# Treinando a MLP
epochs = 1000
learning_rate = 0.1
mlp.train(X_train, y_train, epochs, learning_rate)

# Testando a MLP no conjunto de teste
predictions = mlp.predict(X_test)
accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
print('Accuracy:', accuracy)
