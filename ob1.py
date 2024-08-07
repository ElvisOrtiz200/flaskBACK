import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

df = pd.read_csv('./Obj1.csv')

x = df.drop(columns = 'RiesgoDM2', axis = 1)
y = df['RiesgoDM2']

scaler =StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x_train.shape
x_test.shape
clf = svm.SVC(kernel = 'linear')
clf.fit(x_train, y_train)
x_train_prediction = clf.predict(x_train)
accuracy_score(x_train_prediction, y_train)

x_test_prediction = clf.predict(x_test)
accuracy_score(x_test_prediction, y_test)
entrada_ex = (10,101,76,180,32.9,0.171,63)

entrada_array = np.asarray(entrada_ex)

entrada_array_ref = entrada_array.reshape(1,-1)

std_data = scaler.transform(entrada_array_ref)
prediction = clf.predict(std_data)
if(prediction[0] == 0 ):
    print("No diabetes")
else: 
    print("Diabetes")

