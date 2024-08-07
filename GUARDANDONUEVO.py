import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Cargar el dataset
df = pd.read_csv('./Obj1_clean.csv')

# Separar características y variable objetivo
X = df[['DiabetesGestacional', 'Glucosa', 'PresionArterial', 'Insulina', 'IMC', 'FuncionPedigriDiabetes', 'Edad']]
y = df['RiesgoDM2']

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Guardar el scaler y el modelo
joblib.dump(scaler, 'objetive1scaler.pkl')
joblib.dump(model, 'objetive1model.pkl')
