import joblib
import numpy as np

# Cargar el modelo entrenado y el scaler
model = joblib.load('objetive1model.pkl')
scaler = joblib.load('objetive1scaler.pkl')

# Datos de entrada (ejemplo)
entrada_ex = np.array([
    [1, 115, 70, 96, 34.6, 0.529, 32]
])
# ,
#     [5, 166, 72, 175, 25.8, 0.587, 51],
#     [1, 115, 70, 96, 34.6, 0.529, 32],
#     [11, 143, 94, 146, 36.6, 0.254, 51],
#     [10, 125, 70, 115, 31.1, 0.205, 41]
# Escalar los datos de entrada
entrada_ex_scaled = scaler.transform(entrada_ex)

# Hacer predicciones
predicciones = model.predict(entrada_ex_scaled)

# Mostrar las predicciones
for i, pred in enumerate(predicciones):
    print(f"Entrada {i+1}: {'Diabetes' if pred == 1 else 'No diabetes'}")
