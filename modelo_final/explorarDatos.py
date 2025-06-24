import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración general de gráficos
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def cargar_dataset(ruta='data/dataset_limpio.csv'):
    return pd.read_csv(ruta)

def mostrar_info_general(df):
    print("[ℹ] Primeras filas del dataset:")
    print(df.head())
    print("\n[📊] Descripción estadística:")
    print(df.describe())
    print("\n[🔎] Valores nulos por columna:")
    print(df.isnull().sum())

def graficar_histogramas(df, carpeta='graficos'):
    os.makedirs(carpeta, exist_ok=True)
    for col in df.columns[:-1]:  # Excluye la variable clase
        plt.figure()
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f'Distribución de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        ruta = f"{carpeta}/hist_{col}.png"
        plt.savefig(ruta)
        plt.close()
        print(f"[📈] Histograma guardado: {ruta}")

def graficar_boxplots(df, carpeta='graficos'):
    os.makedirs(carpeta, exist_ok=True)
    for col in df.columns[:-1]:
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot de {col}')
        plt.tight_layout()
        ruta = f"{carpeta}/box_{col}.png"
        plt.savefig(ruta)
        plt.close()
        print(f"[📦] Boxplot guardado: {ruta}")

def graficar_matriz_correlacion(df, carpeta='graficos'):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title("Matriz de correlación")
    plt.tight_layout()
    ruta = f"{carpeta}/matriz_correlacion.png"
    plt.savefig(ruta)
    plt.close()
    print(f"[🧊] Matriz de correlación guardada: {ruta}")

if __name__ == "__main__":
    print("[📥] Cargando dataset limpio...")
    df = cargar_dataset()

    print("[🔍] Mostrando información general del dataset:")
    mostrar_info_general(df)

    print("[📊] Generando histogramas...")
    graficar_histogramas(df)

    print("[📦] Generando boxplots...")
    graficar_boxplots(df)

    print("[🧊] Generando matriz de correlación...")
    graficar_matriz_correlacion(df)

    print("[✅] Análisis exploratorio finalizado.")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración general de gráficos
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def cargar_dataset(ruta='data/dataset_limpio.csv'):
    return pd.read_csv(ruta)

def mostrar_info_general(df):
    print("[ℹ] Primeras filas del dataset:")
    print(df.head())
    print("\n[📊] Descripción estadística:")
    print(df.describe())
    print("\n[🔎] Valores nulos por columna:")
    print(df.isnull().sum())

def graficar_histogramas(df, carpeta='graficos'):
    os.makedirs(carpeta, exist_ok=True)
    for col in df.columns[:-1]:  # Excluye la variable clase
        plt.figure()
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f'Distribución de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        ruta = f"{carpeta}/hist_{col}.png"
        plt.savefig(ruta)
        plt.close()
        print(f"[📈] Histograma guardado: {ruta}")

def graficar_boxplots(df, carpeta='graficos'):
    os.makedirs(carpeta, exist_ok=True)
    for col in df.columns[:-1]:
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot de {col}')
        plt.tight_layout()
        ruta = f"{carpeta}/box_{col}.png"
        plt.savefig(ruta)
        plt.close()
        print(f"[📦] Boxplot guardado: {ruta}")

def graficar_matriz_correlacion(df, carpeta='graficos'):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title("Matriz de correlación")
    plt.tight_layout()
    ruta = f"{carpeta}/matriz_correlacion.png"
    plt.savefig(ruta)
    plt.close()
    print(f"[🧊] Matriz de correlación guardada: {ruta}")

if __name__ == "__main__":
    print("[📥] Cargando dataset limpio...")
    df = cargar_dataset()

    print("[🔍] Mostrando información general del dataset:")
    mostrar_info_general(df)

    print("[📊] Generando histogramas...")
    graficar_histogramas(df)

    print("[📦] Generando boxplots...")
    graficar_boxplots(df)

    print("[🧊] Generando matriz de correlación...")
    graficar_matriz_correlacion(df)

    print("[✅] Análisis exploratorio finalizado.")
