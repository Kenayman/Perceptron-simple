import numpy as np
import matplotlib.pyplot as plt

# Función de pérdida
def loss_function(x1, x2):
    return 10 - np.exp(-(x1**2 + 3*x2**2))

# Gradientes parciales
def gradient(x1, x2):
    df_dx1 = 2 * x1 * np.exp(-(x1**2 + 3*x2**2))
    df_dx2 = 6 * x2 * np.exp(-(x1**2 + 3*x2**2))
    return df_dx1, df_dx2

# Hiperparámetros
learning_rate = 0.1
x1 = 1.0
x2 = 1.0
iterations = 100

# Almacenar historial de valores para graficar
x1_history = []
x2_history = []
loss_history = []

# Descenso del Gradiente
for _ in range(iterations):
    df_dx1, df_dx2 = gradient(x1, x2)
    x1 = x1 - learning_rate * df_dx1
    x2 = x2 - learning_rate * df_dx2
    current_loss = loss_function(x1, x2)
    
    # Registrar valores en el historial
    x1_history.append(x1)
    x2_history.append(x2)
    loss_history.append(current_loss)

# Gráfico de la función de pérdida
x1_vals = np.linspace(-2, 2, 400)
x2_vals = np.linspace(-2, 2, 400)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = loss_function(X1, X2)

plt.figure(figsize=(12, 5))
plt.contour(X1, X2, Z, levels=20, cmap='viridis')
plt.scatter(x1_history, x2_history, c='red', label='Descenso del Gradiente')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Descenso del Gradiente')
plt.legend()
plt.savefig("Gradiente Descendiente")
plt.show()
