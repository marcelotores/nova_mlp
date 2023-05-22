import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Carregando o conjunto de dados Iris
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Codificando as classes usando One-Hot Encoding
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y)

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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

        # Inicializando os pesos
        self.weights_hidden = np.random.uniform(size=(self.num_inputs, self.num_hidden))
        self.weights_output = np.random.uniform(size=(self.num_hidden, self.num_outputs))

        # Inicializando os biases
        self.bias_hidden = np.zeros((1, self.num_hidden))
        self.bias_output = np.zeros((1, self.num_outputs))

    def forward(self, X):
        # Camada oculta
        self.hidden_output = sigmoid(np.dot(X, self.weights_hidden) + self.bias_hidden)

        # Camada de saída
        self.output = sigmoid(np.dot(self.hidden_output, self.weights_output) + self.bias_output)

    def backward(self, X, y, learning_rate):
        # Backpropagation na camada de saída
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        # Backpropagation na camada oculta
        hidden_error = np.dot(output_delta, self.weights_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Atualização dos pesos e biases
        self.weights_output += learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.weights_hidden += learning_rate * np.dot(X.T, hidden_delta)
        self.bias_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        self.loss = []

        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            error = np.mean(np.abs(y - self.output))
            self.loss.append(error)

    def predict(self, X):
        self.forward(X)
        return self.output

# Definindo os parâmetros da MLP
num_inputs = X.shape[1]
num_hidden = 5
num_outputs = y.shape[1]

# Criando uma instância da MLP
mlp = MLP(num_inputs, num_hidden, num_outputs)

# Treinando a MLP
epochs = 1000
learning_rate = 0.1
mlp.train(X_train, y_train, epochs, learning_rate)

# Plotando os erros de treinamento e teste
plt.plot(range(epochs), mlp.loss, 'r-', label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

# Testando a MLP no conjunto de teste
predictions = mlp.predict(X_test)
predictions = np.argmax(predictions, axis=1)
y_test = np.argmax(y_test, axis=1)

# Calculando a acurácia
accuracy = np.mean(predictions == y_test)
print('Accuracy:', accuracy)
