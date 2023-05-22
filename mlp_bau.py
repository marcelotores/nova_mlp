import numpy as np

# Função de ativação sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Função de derivada da sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Classe da MLP
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicialização aleatória dos pesos
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, X):
        # Propagação direta
        self.hidden_layer_activation = sigmoid(np.dot(X, self.weights_input_hidden))
        self.output = sigmoid(np.dot(self.hidden_layer_activation, self.weights_hidden_output))
        return self.output

    def backward(self, X, y, learning_rate):
        # Retropropagação do erro
        error = y - self.output
        output_delta = error * sigmoid_derivative(self.output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer_activation)

        # Atualização dos pesos
        self.weights_hidden_output += self.hidden_layer_activation.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)

# Dataset fornecido anteriormente
X = np.array([
    [0.00234148759345, 2.484356589, 396172138406, 13184.3097039, 0.533112719953, 1.0, 51724.5129095, 2.26348078086, 2.77132864597, 0.00115404970126, 0.591052858553, 0.588400990655, 1.0, 1.0, 0.575196926811, 112.209716643, 0.00586893339572, 52731.8212198, 12873885.5732, 3244547962.24, 1.1757859734, 0.00389183850907, 13183.210839, 5.78451996503e-10],
    [0.0017612463819, 3.57921097038, 547622633699, 15717.3902495, 0.482912651321, 1.0, 61734.8578461, 2.30358288427, 2.88598703609, 0.00101015859196, 0.655669947095, 0.559400297869, 1.0, 1.0, 0.535587818571, 122.730858612, 0.00413639880613, 62856.5988948, 16667976.6742, 4549368771.07, 1.4108370606, 0.00924710051938, 15714.126314, 3.77875227425e-10],
    # Adicione mais exemplos de treinamento aqui...
])

y = np.array([
    [0],
    [1],
    # Adicione mais rótulos aqui...
])

# Normalização dos dados
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Criação da MLP

mlp = MLP(input_size=X.shape[1], hidden_size=16, output_size=1)

# Treinamento da MLP
mlp.train(X, y, epochs=1000, learning_rate=0.1)

# Exemplo de teste
X_test = np.array([[0.0034, 2.2, 456789, 9000, 0.6, 0.9, 12345, 2.1, 2.5, 0.002, 0.4, 0.3, 0.8, 1.0, 0.7, 80, 0.002, 100000, 20000000, 5000000000, 1.3, 0.001, 9000, 1e-10]])
X_test = (X_test - np.mean(X, axis=0)) / np.std(X, axis=0)
predicted_output = mlp.forward(X_test)

print("Predicted output:", predicted_output)