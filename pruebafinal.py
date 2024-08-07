import pandas as pd
import numpy as np
import joblib

# Paso 1: Cargar el modelo y el scaler entrenados
scaler = joblib.load('objetive1scaler.pkl')  # Cargar el StandardScaler guardado
model = joblib.load('objetive1model.pkl')  # Cargar el modelo entrenado

# Paso 2: Preparar los datos de entrada
# Suponiendo que tienes una fila de datos para predecir
entrada_ex = np.array([0,129,110,130,67.1,0.319,26])  # Cambia esto a tus datos reales
entrada_ex = entrada_ex.reshape(1, -1)  # Reshape a (1, -1) para una única muestra

# Paso 3: Escalar los datos de entrada
entrada_ex_scaled = scaler.transform(entrada_ex)

# Paso 4: Hacer la predicción
prediccion = model.predict(entrada_ex_scaled)

# Paso 5: Mostrar la predicción
print("Predicción:", "Diabetes" if prediccion[0] == 1 else "No diabetes")
