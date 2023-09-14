import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generar datos aleatorios para la gráfica 3D
np.random.seed(0)
num_puntos = 100
x = np.random.rand(num_puntos)
y = np.random.rand(num_puntos)
z = np.random.rand(num_puntos)

# Crear una figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Crear la gráfica 3D
ax.scatter(x, y, z, c='b', marker='o')

# Configurar etiquetas de los ejes
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.set_zlabel('Eje Z')

# Mostrar la gráfica
plt.show()
