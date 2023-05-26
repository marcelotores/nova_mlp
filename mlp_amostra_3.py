import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

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
        self.train_accuracies = []
        self.test_accuracies = []
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
            self.peso1.append(self.W1[0][0])


    def train(self, X_train, y_train, X_test, y_test, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            # Forward pass para treinamento
            output_train = self.forward(X_train)
            train_error = np.mean(np.abs(output_train - y_train))
            self.train_errors.append(train_error)
            #
            train_accuracy = np.sum(np.argmax(output_train, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
            self.train_accuracies.append(train_accuracy)

            # Backward pass para treinamento
            self.backward(X_train, y_train, learning_rate)
            # Forward pass para teste
            output_test = self.forward(X_test)
            test_error = np.mean(np.abs(output_test - y_test))
            self.test_errors.append(test_error)
            test_accuracy = np.sum(np.argmax(output_test, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
            self.test_accuracies.append(test_accuracy)
            #print(self.peso1)
            # P e p -> d
            #exit()
        # Calcular a acurácia de treinamento final
        y_train_pred = np.argmax(output_train, axis=1)
        y_train_true = np.argmax(y_train, axis=1)
        train_accuracy = accuracy_score(y_train_true, y_train_pred)
        print("Acurácia de Treinamento Final: {:.2%}".format(train_accuracy))

    def predict(self, X):
        return np.round(self.forward(X))

    def compute_confusion_matrix(self, X_train, y_train):
        predictions = self.predict(X_train)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_train, axis=1)
        return confusion_matrix(y_true, y_pred)


