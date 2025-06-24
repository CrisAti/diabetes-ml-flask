import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import load
import os

def cargar_datos_y_modelos():
    df = pd.read_csv("data/dataset_limpio.csv")
    modelo = load("modelos_guardados/modelo_final.pkl")
    scaler = load("modelos_guardados/escalador.pkl")
    return df, modelo, scaler

def preparar_datos(df, scaler):
    X = df.drop("class", axis=1)
    y = df["class"].map({'tested_negative': 0, 'tested_positive': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, y_test

def graficar_matriz_confusion(y_test, y_pred, carpeta='graficos'):
    os.makedirs(carpeta, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negativa", "Positiva"])
    disp.plot(cmap='Blues')
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    ruta = f"{carpeta}/matriz_confusion.png"
    plt.savefig(ruta)
    plt.close()
    print(f"[🧩] Matriz de confusión guardada: {ruta}")

def graficar_curva_roc(y_test, y_scores, carpeta='graficos'):
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc_score = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})", color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    ruta = f"{carpeta}/curva_roc.png"
    plt.savefig(ruta)
    plt.close()
    print(f"[📉] Curva ROC guardada: {ruta}")

if __name__ == "__main__":
    print("[📥] Cargando datos y modelo...")
    df, modelo, scaler = cargar_datos_y_modelos()

    print("[🔄] Preparando conjunto de prueba...")
    X_test_scaled, y_test = preparar_datos(df, scaler)

    print("[🔍] Generando predicciones...")
    y_pred = modelo.predict(X_test_scaled)

    try:
        y_scores = modelo.predict_proba(X_test_scaled)[:, 1]
    except AttributeError:
        print("[⚠] El modelo no soporta predict_proba(). Se usará decisión binaria para ROC.")
        y_scores = modelo.decision_function(X_test_scaled)

    print("[🧩] Graficando matriz de confusión...")
    graficar_matriz_confusion(y_test, y_pred)

    print("[📉] Graficando curva ROC...")
    graficar_curva_roc(y_test, y_scores)

    print("[✅] Evaluación finalizada.")
