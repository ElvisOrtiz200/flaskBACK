import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib

# Cargar los datos
df = pd.read_csv('./Obj1.csv')

# Separar las características y la etiqueta
x = df.drop(columns='RiesgoDM2', axis=1)
y = df['RiesgoDM2']

# Escalar los datos
scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)

# Dividir los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Entrenar el modelo SVM
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

# Evaluar el modelo (opcional)
x_train_prediction = clf.predict(x_train)
train_accuracy = accuracy_score(x_train_prediction, y_train)
print(f"Train Accuracy: {train_accuracy}")

x_test_prediction = clf.predict(x_test)
test_accuracy = accuracy_score(x_test_prediction, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Guardar el modelo y el escalador
joblib.dump(clf, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')