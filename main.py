from flask import Flask, jsonify, request
import joblib
import numpy as np
from flask_cors import CORS
import pandas as pd
import numpy
# import mysql.connector

app = Flask(__name__)
CORS(app)  # This allows all origins by default

# Load your models and scalers
model = joblib.load('./objetive1model.pkl')
scaler = joblib.load('./objetive1scaler.pkl')

model3 = joblib.load('./ObjetiveCORRECTOmodel.pkl')
scaler3 = joblib.load('./ObjetiveCORRECTOscaler.pkl')

svc = joblib.load('./objetive2model.pkl')
scaler2 = joblib.load('./objetive2scaler.pkl')
le_genero = joblib.load('./label_encoder_genero.pkl')
le_antecedentes = joblib.load('label_encoder_antecedentes.pkl')

# def connect_db():
#     return mysql.connector.connect(
#         host="localhost",
#         user="root",
#         password="123456",
#         database="machine"
#     )

def cadena_a_dataframe(cadena):
    valores = cadena.split(',')
    datos_entrada = {
        'Edad': int(valores[0]),
        'Genero': valores[1],
        'Pesokg': float(valores[2]),
        'IMC': float(valores[3]),
        'InsulinaHistorica': float(valores[4]),
        'HbA1cPercent': float(valores[5]),
        'Glucemiabasal': float(valores[6]),
        'Glucemia2h': float(valores[7]),
        'Antecedentesfamiliares': valores[8]
    }
    return pd.DataFrame([datos_entrada])

# @app.route('/api/pacientesByName', methods=['GET'])
# def buscar_pacientes_byName():
#     paciente = request.args.get('pacienteBuscado')
    
#     conn = connect_db()
#     cursor = conn.cursor(dictionary=True)
#     query = """
#         SELECT id, nombre, apellido, dni, email
#         FROM pacient
#         WHERE nombre LIKE %s OR apellido LIKE %s
#     """
#     cursor.execute(query, (f"%{paciente}%", f"%{paciente}%"))
#     pacientes = cursor.fetchall()
#     cursor.close()
#     conn.close()
    
#     return jsonify(pacientes)

# @app.route('/guardar_datos_1', methods=['POST'])
# def guardar_datos():
#     data = request.json
#     connection = connect_db()
#     cursor = connection.cursor()

#     query = """
#     INSERT INTO datos_diabetes (diabetes_gestacional, glucosa, presion_arterial, insulina, imc, funcion_pedigri, edad)
#     VALUES (%s, %s, %s, %s, %s, %s, %s)
#     """
#     values = (
#         data['diabetesGestacional'],
#         data['glucosa'],
#         data['presionArterial'],
#         data['insulina'],
#         data['imc'],
#         data['funcionPedrigri'],
#         data['edad']
#     )
#     cursor.execute(query, values)
#     connection.commit()
#     cursor.close()
#     connection.close()
#     return jsonify({'message': 'Datos guardados exitosamente'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data['values']).reshape(1, -1)
    std_data = scaler.transform(input_data)
    prediction = model.predict(std_data)
    return jsonify({'prediction': int(prediction[0])})

@app.route('/process_values', methods=['POST'])
def process_values():
    data = request.get_json(force=True)
    cadena = data['cadena']
    # Convertir la cadena de valores en una lista de números
    valores = cadena.split(',')
    valores = [float(v) if '.' in v else int(v) for v in valores]
    # Convertir la lista de valores en un array NumPy
    input_data = np.array([valores])
    # Escalar los datos de entrada usando scaler2
    input_data_scaled = scaler2.transform(input_data)
    # Realizar la predicción usando el modelo svc
    prediction = svc.predict(input_data_scaled)
    # Procesar el último valor
    resultado = valores[-1] * 2
    return jsonify({
        'result': resultado,
        'values_received': valores,
        'prediction': int(prediction[0]),  # Asegúrate de que prediction sea un entero
        'message': 'Enviaste estos valores'
    })


# @app.route('/get_reporte_metformina', methods=['GET'])
# def get_reporte_metformina():
#     connection = mysql.connector.connect(
#         host="localhost",
#         user="root",
#         password="123456",
#         database="machine",
#         port=3306
#     )
    
#     cursor = connection.cursor(dictionary=True)
#     query = """
#     SELECT DATE(medicamento.fecha) AS fecha, SUM(medicamento.prediccion) AS total_prediccion
#     FROM medicamento
#     INNER JOIN pacient ON medicamento.pacient_id = pacient.id
#     WHERE pacient.id = %s
#     GROUP BY DATE(medicamento.fecha)
#     ORDER BY DATE(medicamento.fecha) ASC
#     """
    
#     pacient_id = 1  # Reemplaza esto con el valor adecuado
#     cursor.execute(query, (pacient_id,))
#     rows = cursor.fetchall()
    
#     cursor.close()
#     connection.close()
    
#     return jsonify(rows)





@app.route('/objetive3', methods=['POST'])
def proceso_tres():
    data = request.get_json(force=True)
    cadena = data['cadena']
    # Convertir la cadena de valores en una lista de números
    valores = cadena.split(',')
    valores = [float(v) if '.' in v else int(v) for v in valores]
    # Convertir la lista de valores en un array NumPy
    input_data = np.array([valores])
    # Escalar los datos de entrada usando scaler2
    input_data_scaled = scaler3.transform(input_data)
    # Realizar la predicción usando el modelo svc
    prediction = model3.predict(input_data_scaled)
    # Procesar el último valor
    resultado = valores[-1] * 2
    return jsonify({
        'result': resultado,
        'values_received': valores,
        'prediction': int(prediction[0]),  # Asegúrate de que prediction sea un entero
        'message': 'Enviaste estos valores'
    })

@app.route('/predict_scaled', methods=['POST'])
def predict_scaled():
    data = request.get_json(force=True)
    valores = data['values']
    
    # Convertir la lista de valores en un array de NumPy
    input_data = np.array([valores])
    
    # Escalar los datos de entrada
    input_data_scaled = scaler2.transform(input_data)
    
    # Realizar predicción
    prediction = svc.predict(input_data_scaled)

    return jsonify({
        'prediction': int(prediction[0])
    })


@app.route('/prediction2', methods=['POST'])
def oscarrin():
    input_data = np.array([[46, 0, 79, 25.2, 49.0, 6.6, 125, 191, 1]])
    
    # Escalar los datos de entrada
    input_data_scaled = scaler.transform(input_data)
    
    # Realizar predicción
    prediction = model.predict(input_data_scaled)
    df_entrada_scaled = scaler2.transform(input_data)

    prediccion = svc.predict(df_entrada_scaled)
    prediccion_proba = svc.predict_proba(df_entrada_scaled)

    return jsonify({
        'prediction': int(prediccion[0]),
        'probability': prediccion_proba[0].tolist()
    })

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hola, mundo!'})

@app.route('/multiply', methods=['POST'])
def multiply():
    data = request.get_json(force=True)
    number = data.get('number', 0)
    result = number * 2
    return jsonify({'result': result})
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
