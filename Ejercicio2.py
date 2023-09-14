import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Función para cargar datos desde un archivo CSV
def cargar_datos(nombre_archivo):
    datos = pd.read_csv(nombre_archivo, header=None)
    return datos.values

# Función para entrenar un perceptrón simple
def perceptron_entrenamiento(X, y, tasa_aprendizaje, max_epocas):
    # Inicializar pesos y bias
    num_entradas = X.shape[1]
    pesos = np.random.rand(num_entradas)
    bias = np.random.rand()

    epoca = 0
    while epoca < max_epocas:
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

        epoca += 1

    return pesos, bias

# Función para calcular la precisión
def calcular_precision(X, y, pesos, bias):
    aciertos = 0
    for i in range(X.shape[0]):
        salida = np.dot(X[i], pesos) + bias
        if (salida >= 0 and y[i] == 1) or (salida < 0 and y[i] == 0):
            aciertos += 1
    precision = aciertos / X.shape[0]
    return precision

# Cargar datos desde el archivo CSV
archivos = ["spheres1d10.csv", "spheres2d10.csv", "spheres2d50.csv", "spheres2d70.csv"]

for nombre_archivo in archivos:
    print(f"Usando el archivo {nombre_archivo}:")

    datos = cargar_datos(nombre_archivo)

    # Definir el número de particiones y el porcentaje de datos de entrenamiento
    num_particiones = 10
    porcentaje_entrenamiento = 0.8

    # Crear las particiones
    precisiones = []

    for _ in range(num_particiones):
        np.random.shuffle(datos)
        total_filas = datos.shape[0]
        filas_entrenamiento = int(porcentaje_entrenamiento * total_filas)

        datos_entrenamiento = datos[:filas_entrenamiento]
        datos_prueba = datos[filas_entrenamiento:]

        X_entrenamiento = datos_entrenamiento[:, :-1]  # Entradas
        y_entrenamiento = datos_entrenamiento[:, -1]   # Salidas
        X_prueba = datos_prueba[:, :-1]                # Entradas de prueba
        y_prueba = datos_prueba[:, -1]                 # Salidas de prueba

        # Entrenar el perceptrón con el conjunto de entrenamiento
        tasa_aprendizaje = 0.1
        max_epocas = 100
        pesos_entrenados, bias_entrenado = perceptron_entrenamiento(X_entrenamiento, y_entrenamiento, tasa_aprendizaje, max_epocas)

        # Calcular la precisión en el conjunto de prueba
        precision = calcular_precision(X_prueba, y_prueba, pesos_entrenados, bias_entrenado)
        precisiones.append(precision)

    # Calcular estadísticas resumidas
    promedio_precision = np.mean(precisiones)
    desviacion_estandar_precision = np.std(precisiones)

    # Imprimir resultados
    print(f"Resultados de las {num_particiones} particiones:")
    print(f"Promedio de precisión: {promedio_precision}")
    print(f"Desviación estándar de precisión: {desviacion_estandar_precision}")
    print()
resultados = []

for nombre_archivo in archivos:
    print(f"Usando el archivo {nombre_archivo}:")

    datos = cargar_datos(nombre_archivo)
    
    # Crear una lista para almacenar los resultados de precisión para esta archivo
    precisiones_archivo = []

    # Definir una lista de tamaños de conjunto de entrenamiento para la tercera dimensión
    tamanios_entrenamiento = []

    for _ in range(num_particiones):
        np.random.shuffle(datos)
        total_filas = datos.shape[0]
        filas_entrenamiento = int(porcentaje_entrenamiento * total_filas)

        datos_entrenamiento = datos[:filas_entrenamiento]
        datos_prueba = datos[filas_entrenamiento:]

        X_entrenamiento = datos_entrenamiento[:, :-1]  # Entradas
        y_entrenamiento = datos_entrenamiento[:, -1]   # Salidas
        X_prueba = datos_prueba[:, :-1]                # Entradas de prueba
        y_prueba = datos_prueba[:, -1]                 # Salidas de prueba

        # Entrenar el perceptrón con el conjunto de entrenamiento
        tasa_aprendizaje = 0.1
        max_epocas = 100
        pesos_entrenados, bias_entrenado = perceptron_entrenamiento(X_entrenamiento, y_entrenamiento, tasa_aprendizaje, max_epocas)

        # Calcular la precisión en el conjunto de prueba
        precision = calcular_precision(X_prueba, y_prueba, pesos_entrenados, bias_entrenado)
        precisiones_archivo.append(precision)

        # Almacenar el tamaño del conjunto de entrenamiento para la tercera dimensión
        tamanios_entrenamiento.append(filas_entrenamiento)

    resultados.append((nombre_archivo, tamanios_entrenamiento, precisiones_archivo))

# Representación 3D 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for nombre_archivo, tamanios_entrenamiento, precisiones_archivo in resultados:
    # Obtener la dimensión del archivo sin la extensión ".csv"
    dimension = int(nombre_archivo.split('d')[1].split('.')[0])

    ax.scatter(tamanios_entrenamiento, [dimension]*num_particiones, precisiones_archivo, label=nombre_archivo)

ax.set_xlabel('Tamaño del Conjunto de Entrenamiento')
ax.set_ylabel('Dimensión del Espacio')
ax.set_zlabel('Precisión')
ax.legend()

plt.show()
