import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Función para cargar el conjunto de datos
def load_data(filename):
    data = pd.read_csv(filename)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

# Función para dividir el conjunto de datos en entrenamiento y prueba (80% - 20%)
def split_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Función de activación: sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Clase para la red neuronal
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [np.random.rand(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]

    def feed_forward(self, X):
        self.z_values = []
        self.activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = sigmoid(z)
            self.activations.append(activation)
        return self.activations[-1]

    def backward_propagation(self, X, y, learning_rate):
        delta = (self.activations[-1] - y) * sigmoid_derivative(self.activations[-1])
        deltas = [delta]
        for i in range(len(self.weights)-1, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * sigmoid_derivative(self.activations[i])
            deltas.append(delta)
        deltas.reverse()
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(self.activations[i].T, deltas[i])
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    def train(self, X, y, learning_rate, epochs):
        for _ in range(epochs):
            for i in range(len(X)):
                x = X[i].reshape(1, -1)
                target = y[i].reshape(1, -1)
                self.feed_forward(x)
                self.backward_propagation(x, target, learning_rate)

    def predict(self, X):
        predictions = []
        for x in X:
            x = x.reshape(1, -1)
            prediction = self.feed_forward(x)
            predictions.append(prediction)
        return np.array(predictions).squeeze()

# Cargar el conjunto de datos y dividirlo en entrenamiento y prueba
X, y = load_data('concentlite.csv')
X_train, X_test, y_train, y_test = split_data(X, y)

# Definir la arquitectura de la red neuronal (por ejemplo, 2 capas ocultas con 5 neuronas cada una)
layers = [X_train.shape[1], 5, 5, 1]

# Crear una instancia de la red neuronal
nn = NeuralNetwork(layers)

# Entrenar la red neuronal
learning_rate = 0.1
epochs = 1000
nn.train(X_train, y_train, learning_rate, epochs)

# Predecir en el conjunto de prueba
y_pred = nn.predict(X_test)

# Calcular la precisión
accuracy = np.mean((y_pred > 0.5) == y_test)
print("Precisión en el conjunto de prueba:", accuracy)

# Visualizar los resultados de la clasificación
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm')
plt.title("Clasificación con MLP")
plt.show()
