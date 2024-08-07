import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Cargar el dataset
df = pd.read_csv('./Obj1_clean.csv')

# Verificar datos faltantes y tipos de datos


# Reemplazar valores no numéricos y espacios en blanco con NaN
df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

# Convertir columnas a tipo numérico, forzando los errores a NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Eliminar filas con valores nulos
df.dropna(inplace=True)



# Separar características y variable objetivo
X = df[['DiabetesGestacional', 'Glucosa', 'PresionArterial', 'Insulina', 'IMC', 'FuncionPedigriDiabetes', 'Edad']]
y = df['RiesgoDM2']

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.01, random_state=42)

# Crear y entrenar el modelo SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Hacer predicciones
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluar el modelo


# Predecir con un nuevo ejemplo
entrada_ex = np.array([1,95,82,180,35,0.233,43]).reshape(1, -1)
entrada_ex_scaled = scaler.transform(entrada_ex)
prediccion = model.predict(entrada_ex_scaled)

if prediccion[0] == 0:
    print("No diabetes")
else:
    print("Diabetes")
