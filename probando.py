import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Cargar el dataset de prueba desde un archivo CSV
df_test = pd.read_csv('./DatosPrueba.csv', header=None)

# Verificar datos cargados
print("Datos de prueba:")
print(df_test)

# Cargar el scaler previamente entrenado
scaler = joblib.load('objetive1scaler.pkl')  # Asegúrate de tener el scaler guardado

# Escalar los datos de prueba
X_test_scaled = scaler.transform(df_test)

# Cargar el modelo entrenado
model = joblib.load('objetive1model.pkl')

# Hacer predicciones
predicciones = model.predict(X_test_scaled)

# Mostrar las predicciones
print("\nPredicciones:")
for i, pred in enumerate(predicciones):
    print(1 if pred == 1 else 0)

# Crear un DataFrame solo con la columna de predicciones
df_predicciones = pd.DataFrame(predicciones)

# Guardar solo la columna de predicciones en un archivo CSV
df_predicciones.to_csv('p1.csv', index=False)
