import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata
import re
import tkinter as tk
from tkinter import messagebox

class DataPreprocessing:
    """Clase para la normalización y preprocesamiento de datos."""

    @staticmethod
    def normalize_text(text):
        """Normaliza el texto eliminando caracteres especiales y acentos."""
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
        text = re.sub(r'[^\w\s]', '', text)  # Elimina caracteres especiales
        text = text.lower().strip()  # Convierte a minúsculas y elimina espacios en blanco
        return text

def clasificacion_texto_ia(df):
    """Clasifica textos en el DataFrame utilizando un modelo de IA."""
    if df.empty:
        messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
        return

    if 'texto' not in df.columns or 'categoria' not in df.columns:
        messagebox.showwarning("Advertencia", "El DataFrame debe contener las columnas 'texto' y 'categoria'.")
        return

    X = df['texto'].apply(DataPreprocessing.normalize_text).values
    y = df['categoria'].values

    # Vectorización del texto
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    # Clasificación con Naive Bayes
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = classifier.score(X_test, y_test)

    explanation = (
        f"📝 CLASIFICACIÓN DE TEXTOS CON IA\n\n"
        f"Este análisis lee textos y los clasifica automáticamente en categorías usando IA.\n\n"
        f"🔍 RESULTADO:\n"
        f"   • Precisión del sistema: {accuracy:.2f} (más cerca de 1 = mejor)\n\n"
        f"💡 ¿CÓMO FUNCIONA?\n"
        f"   • La IA aprende qué palabras van con qué categorías\n"
        f"   • Luego clasifica nuevos textos automáticamente\n\n"
        f"📋 USO: Organizar textos agrícolas por temas o tipos"
    )
    messagebox.showinfo("Clasificación de Texto con IA", explanation)

# Ejemplo de uso:
# df = pd.read_csv('tu_archivo.csv')  # Cargar tu DataFrame
# clasificacion_texto_ia(df)  # Ejecutar la función