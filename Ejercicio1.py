#CAMARENA ARGUELLES KENNETH 215720727
import numpy as np
import matplotlib.pyplot as plt


patrones_xor = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])

alteraciones = np.random.uniform(low=-0.05, high=0.05, size=patrones_xor.shape)


patrones_con_alteraciones = patrones_xor + alteraciones

salidas_xor = np.array([1, 0, 0, 1])

datos_xor = np.column_stack((patrones_con_alteraciones, salidas_xor))

# (80% - 20%)
np.random.shuffle(datos_xor)
total_filas = datos_xor.shape[0]
filas_entrenamiento = int(0.8 * total_filas)

entrenamiento = datos_xor[:filas_entrenamiento]
prueba = datos_xor[filas_entrenamiento:]

# Guardar los conjuntos de entrenamiento y prueba en archivos CSV
np.savetxt('OR_trn.csv', entrenamiento, delimiter=',', fmt='%1.2f')
np.savetxt('OR_tst.csv', prueba, delimiter=',', fmt='%1.2f')

# Paso 2: Entrenar el perceptrón simple

def perceptron_entrenamiento(X, y, tasa_aprendizaje, maxGen):
    # Inicializar pesos y bias
    num_entradas = X.shape[1]
    pesos = np.random.rand(num_entradas)
    bias = np.random.rand()

    generacion = 0
    while generacion < maxGen:
        error_total = 0
        for i in range(X.shape[0]):
            # Calcular la salida del perceptrón
            salida = np.dot(X[i], pesos) + bias

            # Aplicar la función de activación (signo)
            if salida >= 0:
                prediccion = 1
            else:
                prediccion = 0

            # Calcular el error
            error = y[i] - prediccion

            # Actualizar pesos y bias
            pesos += tasa_aprendizaje * error * X[i]
            bias += tasa_aprendizaje * error

            error_total += abs(error)

        # Verificar el criterio de finalización
        if error_total == 0:
            break

        generacion += 1

    return pesos, bias

# Entrenar el perceptrón con los datos de entrenamiento XOR
archivo_entrenamiento = 'OR_trn.csv'
datos_entrenamiento = np.genfromtxt(archivo_entrenamiento, delimiter=',')
X_entrenamiento = datos_entrenamiento[:, :-1]  # Entradas
y_entrenamiento = datos_entrenamiento[:, -1]   # Salidas

tasa_aprendizaje = 0.1
maxGen = 100

pesos_entrenados, bias_entrenado = perceptron_entrenamiento(X_entrenamiento, y_entrenamiento, tasa_aprendizaje, maxGen)



clase_0 = X_entrenamiento[y_entrenamiento == 0]
clase_1 = X_entrenamiento[y_entrenamiento == 1]

# Calcular la recta de separación
x_separacion = np.linspace(-1.5, 1.5, 100)
y_separacion = (-pesos_entrenados[0] * x_separacion - bias_entrenado) / pesos_entrenados[1]

# Graficar los patrones y la recta de separación
plt.scatter(clase_0[:, 0], clase_0[:, 1], c='blue', label='Clase 0 (0)')
plt.scatter(clase_1[:, 0], clase_1[:, 1], c='red', label='Clase 1 (1)')
plt.plot(x_separacion, y_separacion, '-', c='green', label='Recta de separación')
plt.xlabel('Entrada 1')
plt.ylabel('Entrada 2')
plt.legend()
plt.title('Patrones XOR y Recta de Separación')
plt.grid(True)
plt.show()
