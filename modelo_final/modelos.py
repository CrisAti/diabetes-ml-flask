import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import os

def cargar_dataset(ruta='data/dataset_limpio.csv'):
    return pd.read_csv(ruta)

def preparar_datos(df):
    X = df.drop('class', axis=1)
    y = df['class'].map({'tested_negative': 0, 'tested_positive': 1})  # Mapeo a 0 y 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def entrenar_modelos(X_train, y_train):
    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "SVM": SVC(probability=True)
    }

    entrenados = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        entrenados[nombre] = modelo
        print(f"[🤖] Modelo entrenado: {nombre}")
    
    return entrenados

def evaluar_modelos(modelos, X_test, y_test):
    resultados = {}

    for nombre, modelo in modelos.items():
        y_pred = modelo.predict(X_test)
        print(f"\n[📋] Evaluación de: {nombre}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=4))

        # Guardar F1-score para comparación posterior
        reporte = classification_report(y_test, y_pred, output_dict=True)
        resultados[nombre] = reporte['weighted avg']['f1-score']

    return resultados

def guardar_mejor_modelo(modelos, f1_scores, scaler, carpeta='modelos_guardados'):
    mejor = max(f1_scores, key=f1_scores.get)
    os.makedirs(carpeta, exist_ok=True)
    dump(modelos[mejor], f'{carpeta}/modelo_final.pkl')
    dump(scaler, f'{carpeta}/escalador.pkl')
    print(f"\n[✅] Modelo guardado: {mejor} (F1-score: {f1_scores[mejor]:.4f})")

if __name__ == "__main__":
    print("[📥] Cargando dataset...")
    df = cargar_dataset()

    print("[🔧] Preparando datos...")
    X_train, X_test, y_train, y_test, scaler = preparar_datos(df)

    print("[🏋️] Entrenando modelos...")
    modelos_entrenados = entrenar_modelos(X_train, y_train)

    print("[📊] Evaluando modelos...")
    f1_scores = evaluar_modelos(modelos_entrenados, X_test, y_test)

    print("[💾] Guardando mejor modelo...")
    guardar_mejor_modelo(modelos_entrenados, f1_scores, scaler)
