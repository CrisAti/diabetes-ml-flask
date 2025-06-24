from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from joblib import load
import os

app = Flask(__name__)

# Cargar modelo y escalador usando rutas absolutas para compatibilidad Azure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELOS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'modelos_guardados'))
modelo = load(os.path.join(MODELOS_DIR, 'modelo_final.pkl'))
scaler = load(os.path.join(MODELOS_DIR, 'escalador.pkl'))

@app.route('/')
def formulario():
    return render_template("index.html")

@app.route('/resultado', methods=["POST"])
def resultado():
    try:
        # Obtener datos del formulario
        entradas = [float(request.form[campo]) for campo in ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']]
        datos_np = np.array(entradas).reshape(1, -1)

        # Escalar y predecir
        datos_escalados = scaler.transform(datos_np)
        prediccion = modelo.predict(datos_escalados)[0]

        mensaje = "✅ No se detecta diabetes tipo 2." if prediccion == 0 else "⚠️ Posible presencia de diabetes tipo 2."
        return render_template("index.html", resultado=mensaje)
    
    except Exception as e:
        return render_template("index.html", resultado=f"❌ Error: {e}")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
