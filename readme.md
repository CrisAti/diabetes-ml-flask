# Predicción de Diabetes Tipo 2 con Machine Learning

Este proyecto implementa un sistema completo para el análisis, preprocesamiento, visualización, modelado y predicción de diabetes tipo 2 a partir de datos clínicos. Incluye una aplicación web interactiva y un notebook explicativo.

## Estructura del Proyecto

- **app/**: Aplicación web Flask para la predicción interactiva.
  - `app.py`: Lógica principal de la app.
  - `static/`: Archivos estáticos (CSS, notebook.html).
  - `templates/`: Plantillas HTML (interfaz de usuario).
- **data/**: Datasets originales y procesados.
- **graficos/**: Gráficos generados durante el análisis exploratorio y la evaluación de modelos.
- **modelo_final/**: Scripts de preprocesamiento, exploración, modelado y evaluación.
- **modelos_guardados/**: Modelos y escaladores entrenados y serializados.
- **notebook/**: Notebook explicativo del flujo completo.

## Instalación y Ejecución

1. **Clona el repositorio y entra a la carpeta del proyecto.**
2. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Ejecuta la aplicación web:**
   ```bash
   cd app
   python app.py
   ```
   Accede a [http://localhost:5000](http://localhost:5000) en tu navegador.

## Flujo de Trabajo

### 1. Preprocesamiento de Datos
- Conversión del dataset original (`.arff`) a CSV.
- Imputación de valores inválidos (ceros) usando la mediana en variables clínicas.

### 2. Análisis Exploratorio
- Generación de histogramas y boxplots para cada variable.
- Cálculo y visualización de la matriz de correlación.

### 3. Modelado y Evaluación
- Entrenamiento de modelos: Regresión Logística, Árbol de Decisión, Random Forest, SVM.
- Selección del mejor modelo según F1-score.
- Evaluación con matriz de confusión y curva ROC.

### 4. Aplicación Web
- Permite ingresar datos clínicos y obtener una predicción inmediata.
- Incluye acceso directo al notebook explicativo en formato HTML.

## Uso del Notebook Explicativo
El notebook (`notebook/notebook.ipynb` y su versión HTML) documenta y explica cada paso del flujo de trabajo, incluyendo código, visualizaciones y conclusiones.

## Créditos
- Desarrollado con Python, Flask, pandas, scikit-learn, matplotlib y seaborn.
- Inspirado en el dataset de diabetes de Pima Indians.

## Licencia
Este proyecto es de uso académico y educativo.
