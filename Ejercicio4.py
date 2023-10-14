import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

# Definición de funciones de activación y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Clase para la red neuronal
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicialización de capas y pesos
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)

    def feedforward(self, X):
        # Capa oculta
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        self.hidden_output = sigmoid(self.hidden_input)

        # Capa de salida
        self.output = sigmoid(np.dot(self.hidden_output, self.weights_hidden_output))

    def backpropagation(self, X, y, learning_rate):
        # Cálculo del error
        error = y - self.output

        # Gradiente en la capa de salida
        delta_output = error * sigmoid_derivative(self.output)

        # Actualización de pesos en la capa de salida
        self.weights_hidden_output += np.dot(self.hidden_output.T, delta_output) * learning_rate

        # Gradiente en la capa oculta
        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * sigmoid_derivative(self.hidden_output)

        # Actualización de pesos en la capa oculta
        self.weights_input_hidden += X.T.dot(delta_hidden) * learning_rate

    def train(self, X, y, learning_rate, epochs):
        for _ in range(epochs):
            self.feedforward(X)
            self.backpropagation(X, y, learning_rate)

    def predict(self, X):
        self.feedforward(X)
        return self.output

# Función para Leave-k-Out
def leave_k_out(X, y, k):
    errors = []

    for i in range(len(X)):
        X_val = X[i]
        y_val = y[i]

        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)

        model = NeuralNetwork(input_size, hidden_size, output_size)
        model.train(X_train, y_train, learning_rate, epochs)

        y_pred = model.predict(X_val)
        y_pred_class = np.argmax(y_pred)
        y_true_class = np.argmax(y_val)

        if y_pred_class != y_true_class:
            errors.append(1)

    return 1 - (sum(errors) / len(X))

# Cargar y preparar los datos
data = np.genfromtxt('irisbin.csv', delimiter=',')
X = data[:, :-3]
y = data[:, -3:]

# Parámetros
input_size = X.shape[1]
hidden_size = 8
output_size = 3
learning_rate = 0.01
epochs = 100

k_out = 5
k_out_accuracy = leave_k_out(X, y, k_out)
print(f'Error Leave-{k_out}-Out: {1 - k_out_accuracy:.2f}')

# Inicializa listas para almacenar los puntos correctamente clasificados y los incorrectamente clasificados
correctly_classified_points = []
incorrectly_classified_points = []

# Realiza Leave-One-Out
for i in range(len(X)):
    X_val = X[i]
    y_val = y[i]

    X_train = np.delete(X, i, axis=0)
    y_train = np.delete(y, i, axis=0)

    model = NeuralNetwork(input_size, hidden_size, output_size)
    model.train(X_train, y_train, learning_rate, epochs)

    y_pred = model.predict(X_val)
    y_pred_class = np.argmax(y_pred)
    y_true_class = np.argmax(y_val)

    if y_pred_class == y_true_class:
        correctly_classified_points.append(X_val)
    else:
        incorrectly_classified_points.append(X_val)

correctly_classified_points = np.array(correctly_classified_points)
incorrectly_classified_points = np.array(incorrectly_classified_points)

# Aplica PCA para reducir a 2D
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Crea un DataFrame para visualizar los resultados
df = pd.DataFrame({'X': X_2d[:, 0], 'Y': X_2d[:, 1], 'Label': ['Correcto' if x in correctly_classified_points else 'Incorrecto' for x in X]})
df['Label'] = pd.Categorical(df['Label'])

# Graficar los puntos
plt.figure(figsize=(8, 6))
colors = {'Correcto': 'g', 'Incorrecto': 'r'}
plt.scatter(df['X'], df['Y'], c=df['Label'].apply(lambda x: colors[x]), marker='o')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Visualización de Resultados Leave-One-Out en 2D')
plt.legend(['Correcto', 'Incorrecto'])
plt.show()
