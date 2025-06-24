import pandas as pd
import numpy as np
from scipy.io import arff
import os

# ==============================
# Funciones principales
# ==============================

def cargar_dataset_arff(ruta_archivo):
    """
    Carga un archivo .arff y lo convierte en DataFrame de pandas.
    """
    data, meta = arff.loadarff(ruta_archivo)
    df = pd.DataFrame(data)

    # Convertir bytes a strings si es necesario
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    return df

def imputar_valores_invalidos(df):
    """
    Reemplaza los ceros inválidos en columnas clínicas por la mediana de la columna.
    """
    columnas_a_imputar = ['plas', 'pres', 'skin', 'insu', 'mass']

    for col in columnas_a_imputar:
        df[col] = df[col].replace(0, np.nan)
        mediana = df[col].median()
        df[col].fillna(mediana, inplace=True)

    return df

def guardar_dataset(df, salida='data/dataset_limpio.csv'):
    """
    Guarda el DataFrame procesado como un archivo CSV.
    """
    os.makedirs(os.path.dirname(salida), exist_ok=True)
    df.to_csv(salida, index=False)
    print(f"[✔] Dataset limpio guardado en: {salida}")

# ==============================
# Ejecución directa
# ==============================

if __name__ == "__main__":
    ruta_entrada = 'data/dataset_37_diabetes.arff'
    print("[📥] Cargando dataset...")
    df = cargar_dataset_arff(ruta_entrada)

    print("[🔧] Imputando valores inválidos...")
    df = imputar_valores_invalidos(df)

    print("[💾] Guardando dataset limpio...")
    guardar_dataset(df)
