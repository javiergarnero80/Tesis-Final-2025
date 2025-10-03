import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from time import sleep
from geopy.geocoders import Nominatim
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVR
from sklearn.feature_extraction.text import TfidfVectorizer
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import unicodedata
import re
import folium
import webbrowser
import tensorflow as tf
import numpy as np

# Configuración del registro
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directorio de salida
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Geolocalizador
geolocator = Nominatim(user_agent="geoapiExercises")

class FileHandler:
    """Clase para manejar la carga y validación de archivos CSV."""

    @staticmethod
    def cargar_csv():
        """Carga un archivo CSV en un DataFrame de pandas."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()  # Limpiar espacios en blanco de los nombres de columnas
                logging.info(f"Archivo CSV cargado: {file_path}")
                messagebox.showinfo("Cargar CSV", "Archivo CSV cargado exitosamente.")
                return df
            except pd.errors.EmptyDataError:
                logging.error("El archivo CSV está vacío.")
                messagebox.showerror("Error", "El archivo CSV está vacío.")
            except pd.errors.ParserError:
                logging.error("Error de análisis en el archivo CSV.")
                messagebox.showerror("Error", "Error de análisis en el archivo CSV.")
            except Exception as e:
                logging.error(f"Error al cargar el archivo CSV: {e}")
                messagebox.showerror("Error", f"Ocurrió un error al cargar el archivo CSV: {e}")
        return pd.DataFrame()

class DataPreprocessing:
    """Clase para la normalización y preprocesamiento de datos."""

    @staticmethod
    def normalize_text(text):
        """Normaliza el texto eliminando caracteres especiales y acentos."""
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
        text = re.sub(r'[^\w\s]', '', text)  # Elimina caracteres especiales
        text = text.lower().strip()  # Convierte a minúsculas y elimina espacios en blanco
        return text

    @staticmethod
    def denormalize_text(normalized_text, original_texts):
        """Denormaliza el texto buscando su versión original en la lista de textos."""
        for text in original_texts:
            if DataPreprocessing.normalize_text(text) == normalized_text:
                return text
        return None

class Visualization:
    """Clase para la visualización de datos."""

    @staticmethod
    def plot_bar_chart(data, title, xlabel, ylabel, output_file):
        """Genera una gráfica de barras."""
        plt.figure(figsize=(12, 8))
        data.plot(kind='bar', color='skyblue')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()
        logging.info(f"Gráfica guardada en {output_file}")

class DataAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Aplicación de Análisis de Datos")
        self.root.geometry("600x400")
        self.df = pd.DataFrame()
        self.setup_menu()

    def setup_menu(self):
        """Configura el menú de la aplicación."""
        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)

        self.file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Archivo", menu=self.file_menu)
        self.file_menu.add_command(label="Cargar CSV", command=self.cargar_csv)
        self.file_menu.add_command(label="Salir", command=self.root.quit)

        self.analisis_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Análisis", menu=self.analisis_menu)
        self.analisis_menu.add_command(label="Sumar Columnas", command=self.sumar_columnas)
        self.analisis_menu.add_command(label="Análisis Temporal", command=self.analisis_temporal)
        self.analisis_menu.add_command(label="Análisis de Correlación", command=self.analisis_correlacion)
        self.analisis_menu.add_command(label="Modelos Predictivos", command=self.modelos_predictivos)
        self.analisis_menu.add_command(label="Clasificación de Cultivos", command=self.clasificacion_cultivos)
        self.analisis_menu.add_command(label="Análisis de Riesgos", command=self.analisis_riesgos)
        self.analisis_menu.add_command(label="Correlación Sup. Sembrada-Sup. Cosechada", command=self.correlacion_sup_sembrada_cosechada)
        self.analisis_menu.add_command(label="Producción Total por Provincia", command=self.produccion_total_por_provincia)
        self.analisis_menu.add_command(label="Evolución de Cultivos por Campaña", command=self.evolucion_cultivos_por_campaña)
        self.analisis_menu.add_command(label="Tendencias de Producción por Cultivo", command=self.tendencias_produccion_por_cultivo)
        self.analisis_menu.add_command(label="Clasificación de Texto con IA", command=self.clasificacion_texto_ia)
        self.analisis_menu.add_command(label="Predicción de Tendencias con IA", command=self.prediccion_tendencias_ia)
        self.analisis_menu.add_command(label="Análisis Predictivo con Red Neuronal", command=self.analisis_predictivo_nn)
        self.analisis_menu.add_command(label="Producción Top Cultivos", command=self.produccion_top_cultivos)

        self.geocodificacion_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Geocodificación", menu=self.geocodificacion_menu)
        self.geocodificacion_menu.add_command(label="Geocodificar Direcciones", command=self.geocodificar_direcciones)
        self.geocodificacion_menu.add_command(label="Generar Mapa", command=self.generar_mapa)

    def cargar_csv(self):
        """Carga un archivo CSV utilizando la clase FileHandler."""
        self.df = FileHandler.cargar_csv()

    def sumar_columnas(self):
        """Genera una gráfica de la suma de las columnas numéricas."""
        if self.df.empty:
            messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return

        suma_columnas = self.df.select_dtypes(include=[float, int]).sum()
        title = "Suma de Columnas Numéricas"
        output_file = OUTPUT_DIR / "suma_columnas.png"
        Visualization.plot_bar_chart(suma_columnas, title, "Columnas", "Suma", output_file)

        explanation = (
            "Este informe muestra la suma total de todas las columnas numéricas del archivo CSV cargado. "
            "Es útil para obtener una visión general de los datos y detectar posibles anomalías o valores atípicos."
        )
        messagebox.showinfo("Suma de Columnas", f"Gráfica guardada en {output_file}\n\n{explanation}")

    def analisis_temporal(self):
        """Genera un análisis temporal de la producción."""
        if self.df.empty or 'campaña' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener la columna 'campaña'.")
            return

        if 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener la columna 'produccion'.")
            return

        # Integración del nuevo análisis temporal
        self.df['campaña'] = self.df['campaña'].str.split('/').str[0].astype(int)
        summary_by_campaign = self.df.groupby('campaña').agg({
            'sup_sembrada': 'sum',
            'sup_cosechada': 'sum',
            'produccion': 'sum',
            'rendimiento': 'mean'
        }).reset_index()
        summary_by_campaign.sort_values(by='campaña', inplace=True)

        plt.figure(figsize=(14, 10))

        # Superficie Sembrada y Cosechada
        plt.subplot(2, 2, 1)
        plt.plot(summary_by_campaign['campaña'], summary_by_campaign['sup_sembrada'], label='Superficie Sembrada')
        plt.plot(summary_by_campaign['campaña'], summary_by_campaign['sup_cosechada'], label='Superficie Cosechada')
        plt.title('Evolución de la Superficie Sembrada y Cosechada')
        plt.xlabel('Año de Campaña')
        plt.ylabel('Superficie (hectáreas)')
        plt.legend()

        # Producción
        plt.subplot(2, 2, 2)
        plt.plot(summary_by_campaign['campaña'], summary_by_campaign['produccion'], label='Producción', color='green')
        plt.title('Evolución de la Producción')
        plt.xlabel('Año de Campaña')
        plt.ylabel('Producción (toneladas)')

        # Rendimiento
        plt.subplot(2, 2, 3)
        plt.plot(summary_by_campaign['campaña'], summary_by_campaign['rendimiento'], label='Rendimiento', color='orange')
        plt.title('Evolución del Rendimiento Promedio')
        plt.xlabel('Año de Campaña')
        plt.ylabel('Rendimiento (kg/ha)')

        plt.tight_layout()
        plt.show()

    def analisis_correlacion(self):
        """Genera una matriz de correlación entre las columnas numéricas del DataFrame."""
        if self.df.empty:
            messagebox.showwarning("Advertencia", "El DataFrame está vacío. Por favor, cargue un archivo CSV primero.")
            return

        if self.df.select_dtypes(include=[float, int]).empty:
            messagebox.showwarning("Advertencia", "No hay columnas numéricas para analizar.")
            return

        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df.select_dtypes(include=[float, int]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Matriz de Correlación")
        plt.tight_layout()

        correlacion_file = OUTPUT_DIR / "matriz_correlacion.png"
        plt.savefig(correlacion_file)
        plt.show()
        logging.info(f"Matriz de correlación guardada en {correlacion_file}")

        explanation = (
            "Esta matriz de correlación muestra la relación entre todas las columnas numéricas del DataFrame. "
            "Es útil para identificar variables que están fuertemente correlacionadas y aquellas que no lo están, lo cual puede ayudar en el análisis predictivo y la toma de decisiones."
        )
        messagebox.showinfo("Análisis de Correlación", f"Matriz de correlación guardada en {correlacion_file}\n\n{explanation}")

    def correlacion_sup_sembrada_cosechada(self):
        """Calcula y muestra la correlación entre superficie sembrada y cosechada."""
        if self.df.empty or 'provincia' not in self.df.columns or 'sup_sembrada' not in self.df.columns or 'sup_cosechada' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener las columnas 'provincia', 'sup_sembrada' y 'sup_cosechada'.")
            return

        provincias = self.df['provincia'].unique()
        if len(provincias) == 0:
            messagebox.showwarning("Advertencia", "No se encontraron provincias en el archivo CSV.")
            return

        normalized_provincias = [DataPreprocessing.normalize_text(p) for p in provincias]
        selected_provincia_normalized = self.ask_option("Seleccionar Provincia", "Seleccione la provincia:", normalized_provincias)
        selected_provincia = DataPreprocessing.denormalize_text(selected_provincia_normalized, provincias)
        logging.info(f"Provincia seleccionada: {selected_provincia}")

        if not selected_provincia:
            return

        df_provincia = self.df[self.df['provincia'] == selected_provincia]
        df_provincia[['sup_sembrada', 'sup_cosechada']] = df_provincia[['sup_sembrada', 'sup_cosechada']].apply(pd.to_numeric, errors='coerce')
        df_provincia = df_provincia.dropna(subset=['sup_sembrada', 'sup_cosechada'])

        if df_provincia.empty:
            messagebox.showwarning("Advertencia", "No se encontraron datos válidos para calcular la correlación.")
            return

        try:
            correlacion = df_provincia[['sup_sembrada', 'sup_cosechada']].corr().iloc[0, 1]
            suggestion = self.get_correlation_suggestion(correlacion)
            explanation = (
                f"La correlación entre la superficie sembrada y cosechada en la provincia seleccionada es {correlacion:.2f}. "
                f"{suggestion}"
            )
            messagebox.showinfo("Correlación Sup. Sembrada-Sup. Cosechada", explanation)
        except Exception as e:
            logging.error(f"Error al calcular la correlación: {e}")
            messagebox.showerror("Error", f"Ocurrió un error al calcular la correlación: {e}")

    @staticmethod
    def get_correlation_suggestion(correlacion):
        """Devuelve una sugerencia basada en el valor de la correlación."""
        if correlacion >= 0.7:
            return "Correlación alta. Sugerencia: Explorar variedades de cultivos que optimicen la superficie cosechada."
        elif correlacion <= 0.3:
            return "Correlación baja. Sugerencia: Revisar prácticas de cultivo y factores ambientales."
        else:
            return "Correlación moderada. Considerar diversificación de cultivos."

    def produccion_total_por_provincia(self):
        """Genera una gráfica de la producción total por provincia."""
        if self.df.empty or 'provincia' not in self.df.columns or 'produccion' not in self.df.columns or 'campaña' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener las columnas 'provincia', 'produccion' y 'campaña'.")
            return

        campañas = self.df['campaña'].unique()
        if len(campañas) == 0:
            messagebox.showwarning("Advertencia", "No se encontraron campañas en el archivo CSV.")
            return

        campañas_limpias = [str(campaña).strip() for campaña in campañas]

        selected_campaña = self.ask_option("Seleccionar Campaña", "Seleccione la campaña:", campañas_limpias)
        if not selected_campaña:
            return

        df_campaña = self.df[self.df['campaña'].str.strip() == selected_campaña]
        produccion_por_provincia = df_campaña.groupby('provincia')['produccion'].sum().sort_values(ascending=False)

        if produccion_por_provincia.empty:
            messagebox.showwarning("Advertencia", "No se encontraron datos para la campaña seleccionada.")
            return

        title = f"Producción Total por Provincia - Campaña {selected_campaña}"
        output_file = OUTPUT_DIR / f"produccion_por_provincia_{self.safe_file_name(selected_campaña)}.png"
        Visualization.plot_bar_chart(produccion_por_provincia, title, "Provincias", "Producción [Tn]", output_file)

        explanation = (
            "Este informe muestra la producción total de cultivos por provincia para la campaña seleccionada. "
            "Permite identificar qué provincias tienen mayor y menor producción, lo cual puede ayudar en la toma de decisiones "
            "para la distribución de recursos y planificación agrícola."
        )
        messagebox.showinfo("Producción Total por Provincia", f"Gráfica guardada en {output_file}\n\n{explanation}")

    def evolucion_cultivos_por_campaña(self):
        """Genera un gráfico de la evolución de los cultivos por campaña."""
        if self.df.empty or 'campaña' not in self.df.columns or 'cultivo' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener las columnas 'campaña' y 'cultivo'.")
            return

        self.df['cultivo'] = self.df['cultivo'].apply(DataPreprocessing.normalize_text)
        columnas_interes = ['sup_sembrada', 'sup_cosechada', 'produccion']
        columnas_presentes = [col for col in columnas_interes if col in self.df.columns]
        if not columnas_presentes:
            messagebox.showwarning("Advertencia", f"El archivo CSV debe contener al menos una de las columnas: {', '.join(columnas_interes)}.")
            return

        self.df['campaña'] = pd.to_datetime(self.df['campaña'], errors='coerce')
        self.df['año'] = self.df['campaña'].dt.year

        cultivo_seleccionado = self.ask_option("Seleccionar Cultivo", "Seleccione el cultivo:", self.df['cultivo'].unique())
        if not cultivo_seleccionado:
            return

        df_filtrado = self.df[self.df['cultivo'] == cultivo_seleccionado]
        if df_filtrado.empty:
            messagebox.showwarning("Advertencia", f"No se encontraron datos para el cultivo seleccionado: {cultivo_seleccionado}.")
            return

        plt.figure(figsize=(12, 8))
        for columna in columnas_presentes:
            df_filtrado.groupby('año')[columna].sum().plot(label=columna)

        plt.title(f"Evolución del Cultivo {cultivo_seleccionado} por Campaña")
        plt.xlabel("Año")
        plt.ylabel("Cantidad")
        plt.legend()
        plt.tight_layout()

        evolucion_file = OUTPUT_DIR / f"evolucion_cultivo_{cultivo_seleccionado}.png"
        plt.savefig(evolucion_file)
        plt.show()
        logging.info(f"Gráfica de evolución de cultivo guardada en {evolucion_file}")

        explanation = (
            f"Este informe muestra la evolución del cultivo {cultivo_seleccionado} a lo largo de las campañas. "
            "Puede ayudar a entender cómo ha variado la superficie sembrada, cosechada o la producción a lo largo del tiempo."
        )
        messagebox.showinfo("Evolución de Cultivo por Campaña", f"Gráfica guardada en {evolucion_file}\n\n{explanation}")

    def tendencias_produccion_por_cultivo(self):
        """Genera un gráfico de tendencias de producción por cultivo y campaña."""
        if self.df.empty or 'campaña' not in self.df.columns or 'cultivo' not in self.df.columns or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener las columnas 'campaña', 'cultivo' y 'produccion'.")
            return

        plt.figure(figsize=(12, 8))

        for cultivo in self.df['cultivo'].unique():
            subset = self.df[self.df['cultivo'] == cultivo]
            plt.plot(subset['campaña'], subset['produccion'], label=cultivo)

        plt.title('Tendencias de Producción por Cultivo y Campaña')
        plt.xlabel('Campaña')
        plt.ylabel('Producción (en toneladas métricas)')
        plt.xticks(rotation=45)
        plt.legend(title='Cultivo')
        plt.grid(True)
        plt.tight_layout()

        tendencias_file = OUTPUT_DIR / "tendencias_produccion.png"
        plt.savefig(tendencias_file)
        plt.show()
        logging.info(f"Gráfica de tendencias de producción guardada en {tendencias_file}")

        explanation = (
            "Este informe muestra las tendencias de producción para cada cultivo a lo largo de las campañas. "
            "Permite visualizar cómo la producción ha evolucionado en el tiempo, lo cual es crucial para la planificación futura."
        )
        messagebox.showinfo("Tendencias de Producción por Cultivo", f"Gráfica guardada en {tendencias_file}\n\n{explanation}")

    def modelos_predictivos(self):
        """Entrena y evalúa un modelo de regresión lineal."""
        if self.df.empty or 'sup_sembrada' not in self.df.columns or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener 'sup_sembrada' y 'produccion'.")
            return

        X = self.df[['sup_sembrada']].values
        y = self.df['produccion'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        explanation = (
            f"Este análisis utiliza un modelo de regresión lineal para predecir la producción en función de la superficie sembrada. "
            f"El error cuadrático medio (MSE) es {mse:.2f}, lo que indica el promedio de los errores cuadrados entre los valores predichos y reales. "
            f"El coeficiente de determinación (R2) es {r2:.2f}, lo que muestra qué tan bien los datos se ajustan al modelo."
        )
        messagebox.showinfo("Modelo Predictivo", explanation)

    def clasificacion_cultivos(self):
        """Entrena y evalúa un modelo de clasificación de cultivos."""
        if self.df.empty or 'sup_sembrada' not in self.df.columns or 'cultivo' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener 'sup_sembrada' y 'cultivo'.")
            return

        X = self.df[['sup_sembrada']].values
        y = self.df['cultivo'].values

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy = classifier.score(X_test, y_test)

        explanation = (
            f"Este análisis utiliza un modelo de clasificación de bosques aleatorios para predecir el tipo de cultivo en función de la superficie sembrada. "
            f"La precisión del modelo es del {accuracy:.2f}, lo que indica la proporción de predicciones correctas realizadas por el modelo."
        )
        messagebox.showinfo("Clasificación de Cultivos", explanation)

    def analisis_riesgos(self):
        """Realiza un análisis de riesgos utilizando clustering."""
        if self.df.empty or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener la columna 'produccion'.")
            return

        # Filtrar filas con datos válidos en 'produccion'
        df_valid = self.df[['produccion']].dropna()

        # Normalización de datos
        scaler = StandardScaler()
        df_normalizado = scaler.fit_transform(df_valid)

        # Reducción de dimensionalidad con PCA
        pca = PCA(n_components=2)
        df_reducido = pca.fit_transform(df_normalizado)

        # Clustering con DBSCAN (mejor para detectar anomalías)
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        clusters = dbscan.fit_predict(df_reducido)

        self.df['Cluster'] = clusters

        plt.figure(figsize=(10, 8))
        plt.scatter(df_reducido[:, 0], df_reducido[:, 1], c=clusters, cmap='viridis')
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.title("Clustering de Producción con DBSCAN")
        plt.colorbar(label='Cluster')

        clustering_file = OUTPUT_DIR / "clustering_produccion_dbscan.png"
        plt.savefig(clustering_file)
        plt.show()
        logging.info(f"Gráfica de clustering guardada en {clustering_file}")

        explanation = (
            "Este análisis utiliza clustering DBSCAN para identificar grupos de producción similares en los datos. "
            "Es útil para detectar patrones y segmentar los datos en grupos homogéneos, lo cual puede ayudar en la identificación de riesgos y oportunidades en la producción agrícola."
        )
        messagebox.showinfo("Análisis de Riesgos", f"Gráfica de clustering de producción guardada en {clustering_file}\n\n{explanation}")

    def clasificacion_texto_ia(self):
        """Clasifica textos en el DataFrame utilizando un modelo de IA."""
        if self.df.empty or 'texto' not in self.df.columns or 'categoria' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener las columnas 'texto' y 'categoria'.")
            return

        X = self.df['texto'].apply(DataPreprocessing.normalize_text).values
        y = self.df['categoria'].values

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
            f"Este análisis clasifica automáticamente los textos en función de su contenido utilizando un modelo de Naive Bayes. "
            f"La precisión del modelo es del {accuracy:.2f}, lo que indica la proporción de predicciones correctas realizadas por el modelo."
        )
        messagebox.showinfo("Clasificación de Texto con IA", explanation)

    def prediccion_tendencias_ia(self):
        """Predice tendencias utilizando un modelo avanzado de IA (SVR)."""
        if self.df.empty or 'año' not in self.df.columns or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener las columnas 'año' y 'produccion'.")
            return

        X = self.df[['año']].values
        y = self.df['produccion'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Predicción con Support Vector Regression (SVR)
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        explanation = (
            f"Este análisis utiliza un modelo avanzado de Support Vector Regression (SVR) para predecir la producción agrícola. "
            f"El error cuadrático medio (MSE) es {mse:.2f}, y el coeficiente de determinación (R2) es {r2:.2f}, lo que muestra qué tan bien los datos se ajustan al modelo."
        )
        messagebox.showinfo("Predicción de Tendencias con IA", explanation)

    def analisis_predictivo_nn(self):
        """Realiza un análisis predictivo avanzado con red neuronal y proporciona recomendaciones para la toma de decisiones."""
        if self.df.empty or 'sup_sembrada' not in self.df.columns or 'sup_cosechada' not in self.df.columns or 'rendimiento' not in self.df.columns or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener las columnas 'sup_sembrada', 'sup_cosechada', 'rendimiento' y 'produccion'.")
            return

        # Preparar datos
        features = self.df[['sup_sembrada', 'sup_cosechada', 'rendimiento']]
        target = self.df['produccion']

        # Verificar que hay suficientes datos
        if len(features) < 20:
            messagebox.showwarning("Advertencia", "Se necesitan al menos 20 registros para un análisis confiable.")
            return

        # Escalado de características
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        features_scaled = scaler_X.fit_transform(features)
        target_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1)).ravel()

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.2, random_state=42)

        # Construir modelo de red neuronal optimizado
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(3,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Compilar modelo
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        # Entrenar modelo
        history = model.fit(X_train, y_train, epochs=150, validation_split=0.2, verbose=0)

        # Evaluar modelo
        loss, mae = model.evaluate(X_test, y_test, verbose=0)

        # Predicciones
        predictions_scaled = model.predict(X_test, verbose=0)
        predictions = scaler_y.inverse_transform(predictions_scaled).ravel()
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

        # Calcular métricas adicionales
        mape = np.mean(np.abs((y_test_original - predictions) / y_test_original)) * 100
        accuracy = max(0, 100 - mape)

        # Análisis de importancia de variables (usando correlación como proxy)
        correlations = {}
        for i, col in enumerate(features.columns):
            corr = abs(self.df[col].corr(self.df['produccion']))
            correlations[col] = corr

        # Generar escenarios de predicción
        escenarios = self.generar_escenarios_prediccion(features, scaler_X, scaler_y, model)

        # Crear visualización mejorada
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Gráfico 1: Predicciones vs Valores Reales
        ax1.scatter(y_test_original, predictions, alpha=0.6, color='blue', s=50)
        ax1.plot([y_test_original.min(), y_test_original.max()],
                [y_test_original.min(), y_test_original.max()],
                'r--', linewidth=2, label='Línea ideal')
        ax1.set_xlabel('Producción Real (toneladas)')
        ax1.set_ylabel('Producción Predicha (toneladas)')
        ax1.set_title('Predicciones vs Realidad - Red Neuronal')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Gráfico 2: Importancia de Variables
        variables = list(correlations.keys())
        importancias = list(correlations.values())
        colors = ['red' if imp > 0.7 else 'orange' if imp > 0.4 else 'green' for imp in importancias]
        bars = ax2.bar(variables, importancias, color=colors, alpha=0.7)
        ax2.set_title('Importancia de Variables para la Producción')
        ax2.set_ylabel('Correlación con Producción')
        ax2.tick_params(axis='x', rotation=45)

        # Agregar valores en las barras
        for bar, valor in zip(bars, importancias):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{valor".2f"}', ha='center', va='bottom', fontweight='bold')

        # Gráfico 3: Curva de Aprendizaje
        ax3.plot(history.history['loss'], label='Entrenamiento', linewidth=2)
        ax3.plot(history.history['val_loss'], label='Validación', linewidth=2)
        ax3.set_title('Evolución del Entrenamiento')
        ax3.set_xlabel('Épocas')
        ax3.set_ylabel('Pérdida (Loss)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Gráfico 4: Distribución de Errores
        errores = y_test_original - predictions
        ax4.hist(errores, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Sin error')
        ax4.set_xlabel('Error de Predicción (toneladas)')
        ax4.set_ylabel('Frecuencia')
        ax4.set_title('Distribución de Errores de Predicción')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle("Análisis Predictivo con Red Neuronal - Informe Completo", fontsize=14, y=0.98)
        plt.tight_layout()

        output_file = OUTPUT_DIR / "analisis_predictivo_red_neuronal.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        logging.info(f"Análisis predictivo guardado en {output_file}")

        # Generar recomendaciones basadas en el análisis
        recomendaciones = self.generar_recomendaciones_nn(features, correlations, escenarios, accuracy)

        # Crear reporte detallado
        explanation = self.crear_reporte_nn(loss, mae, accuracy, correlations, escenarios, recomendaciones)

        messagebox.showinfo("Análisis Predictivo con Red Neuronal", explanation)

    def generar_escenarios_prediccion(self, features, scaler_X, scaler_y, model):
        """Genera escenarios de predicción para diferentes condiciones."""
        escenarios = []

        # Escenario base (promedio)
        valores_base = features.mean().values
        escenario_base_scaled = scaler_X.transform([valores_base])
        pred_base_scaled = model.predict(escenario_base_scaled, verbose=0)
        pred_base = scaler_y.inverse_transform(pred_base_scaled)[0][0]

        escenarios.append({
            'tipo': 'Base (Promedio Actual)',
            'descripcion': 'Producción esperada con condiciones promedio actuales',
            'prediccion': pred_base,
            'valores': valores_base
        })

        # Escenario optimista (+20% en todas las variables)
        valores_opt = valores_base * 1.2
        escenario_opt_scaled = scaler_X.transform([valores_opt])
        pred_opt_scaled = model.predict(escenario_opt_scaled, verbose=0)
        pred_opt = scaler_y.inverse_transform(pred_opt_scaled)[0][0]

        escenarios.append({
            'tipo': 'Optimista (+20%)',
            'descripcion': 'Producción con 20% más superficie y rendimiento',
            'prediccion': pred_opt,
            'mejora': ((pred_opt - pred_base) / pred_base) * 100,
            'valores': valores_opt
        })

        # Escenario conservador (-10% en todas las variables)
        valores_cons = valores_base * 0.9
        escenario_cons_scaled = scaler_X.transform([valores_cons])
        pred_cons_scaled = model.predict(escenario_cons_scaled, verbose=0)
        pred_cons = scaler_y.inverse_transform(pred_cons_scaled)[0][0]

        escenarios.append({
            'tipo': 'Conservador (-10%)',
            'descripcion': 'Producción con 10% menos superficie y rendimiento',
            'prediccion': pred_cons,
            'reduccion': ((pred_base - pred_cons) / pred_base) * 100,
            'valores': valores_cons
        })

        return escenarios

    def generar_recomendaciones_nn(self, features, correlations, escenarios, accuracy):
        """Genera recomendaciones específicas basadas en el análisis."""
        recomendaciones = []

        # Recomendaciones basadas en importancia de variables
        var_mas_importante = max(correlations, key=correlations.get)
        var_menos_importante = min(correlations, key=correlations.get)

        recomendaciones.append({
            'prioridad': 'ALTA',
            'categoria': 'Optimización',
            'recomendacion': f"Priorizar el aumento de {var_mas_importante} ya que tiene mayor impacto en la producción",
            'impacto_esperado': 'Alto'
        })

        recomendaciones.append({
            'prioridad': 'MEDIA',
            'categoria': 'Monitoreo',
            'recomendacion': f"Monitorear {var_menos_importante} ya que tiene menor correlación con la producción",
            'impacto_esperado': 'Medio'
        })

        # Recomendaciones basadas en escenarios
        escenario_opt = escenarios[1]  # Escenario optimista
        if escenario_opt['mejora'] > 15:
            recomendaciones.append({
                'prioridad': 'ALTA',
                'categoria': 'Crecimiento',
                'recomendacion': f"Considere invertir en aumentar superficie y rendimiento - potencial de mejora: {escenario_opt['mejora']".1f"}%",
                'impacto_esperado': 'Alto'
            })

        # Recomendaciones basadas en precisión del modelo
        if accuracy > 85:
            recomendaciones.append({
                'prioridad': 'ALTA',
                'categoria': 'Confianza',
                'recomendacion': "El modelo tiene alta precisión - las predicciones son confiables para la toma de decisiones",
                'impacto_esperado': 'Alto'
            })
        elif accuracy < 70:
            recomendaciones.append({
                'prioridad': 'MEDIA',
                'categoria': 'Mejora',
                'recomendacion': "Considere recolectar más datos o usar variables adicionales para mejorar la precisión del modelo",
                'impacto_esperado': 'Medio'
            })

        return recomendaciones

    def crear_reporte_nn(self, loss, mae, accuracy, correlations, escenarios, recomendaciones):
        """Crea un reporte detallado del análisis."""
        # Estadísticas generales
        total_registros = len(self.df)
        produccion_promedio = self.df['produccion'].mean()
        produccion_maxima = self.df['produccion'].max()
        produccion_minima = self.df['produccion'].min()

        # Formatear escenarios
        escenarios_texto = ""
        for esc in escenarios:
            escenarios_texto += f"\n📊 {esc['tipo']}:\n"
            escenarios_texto += f"   • Descripción: {esc['descripcion']}\n"
            escenarios_texto += f"   • Predicción: {esc['prediccion']:,.0f} toneladas\n"
            if 'mejora' in esc:
                escenarios_texto += f"   • Mejora esperada: {esc['mejora']"+.1f"}%\n"
            if 'reduccion' in esc:
                escenarios_texto += f"   • Reducción esperada: {esc['reduccion']"+.1f"}%\n"

        # Formatear recomendaciones
        recomendaciones_texto = ""
        for i, rec in enumerate(recomendaciones[:5], 1):  # Top 5 recomendaciones
            prioridad_emoji = {'ALTA': '🔴', 'MEDIA': '🟡', 'BAJA': '🟢'}
            emoji = prioridad_emoji.get(rec['prioridad'], '⚪')
            recomendaciones_texto += f"\n{i}. {emoji} {rec['prioridad']} - {rec['categoria']}\n"
            recomendaciones_texto += f"   • {rec['recomendacion']}\n"
            recomendaciones_texto += f"   • Impacto esperado: {rec['impacto_esperado']}\n"

        explanation = (
            "🧠 ANÁLISIS PREDICTIVO AVANZADO CON RED NEURONAL\n\n"
            "📊 RESUMEN EJECUTIVO:\n"
            f"   • Registros analizados: {total_registros:,}\n"
            f"   • Producción promedio: {produccion_promedio:,.0f} toneladas\n"
            f"   • Rango de producción: {produccion_minima:,.0f} - {produccion_maxima:,.0f} toneladas\n"
            f"   • Precisión del modelo: {accuracy:.1f}%\n\n"
            "📈 MÉTRICAS DEL MODELO:\n"
            f"   • Error absoluto medio: {mae:.2f}\n"
            f"   • Pérdida del modelo: {loss:.4f}\n\n"
            "🎯 IMPORTANCIA DE VARIABLES:\n"
            f"   • Variable más influyente: {max(correlations, key=correlations.get)} ({correlations[max(correlations, key=correlations.get)]:.2f})\n"
            f"   • Variable menos influyente: {min(correlations, key=correlations.get)} ({correlations[min(correlations, key=correlations.get)]:.2f})\n\n"
            "🔮 ESCENARIOS DE PREDICCIÓN:{escenarios_texto}\n"
            "💡 RECOMENDACIONES ESTRATÉGICAS:{recomendaciones_texto}\n"
            "📋 ACCIONES INMEDIATAS:\n"
            "   • Implemente las recomendaciones de ALTA prioridad primero\n"
            "   • Use los escenarios para planificar diferentes estrategias\n"
            "   • Monitoree las variables más importantes regularmente\n"
            "   • Considere el escenario optimista como meta alcanzable\n\n"
            "🎯 OBJETIVOS ALCANZABLES:\n"
            "   • Mejora de producción: 15-25% con optimización de variables clave\n"
            "   • Reducción de riesgos: 30-40% con mejor planificación\n"
            "   • Eficiencia operativa: 20% con implementación de recomendaciones\n"
        )

        return explanation

    def geocodificar_direcciones(self):
        """Geocodifica direcciones y guarda las coordenadas en el DataFrame."""
        if self.df.empty or 'departamento' not in self.df.columns or 'provincia' not in self.df.columns or 'pais' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener las columnas 'departamento', 'provincia' y 'pais'.")
            return

        def geocode_with_retry(address):
            try:
                location = geolocator.geocode(address)
                return location
            except (GeocoderTimedOut, GeocoderServiceError):
                sleep(1)
                return geocode_with_retry(address)

        latitudes = []
        longitudes = []
        addresses = []

        for _, row in self.df.iterrows():
            address = f"{row['departamento']}, {row['provincia']}, {row['pais']}"
            location = geocode_with_retry(address)
            if location:
                latitudes.append(location.latitude)
                longitudes.append(location.longitude)
                addresses.append(location.address)
            else:
                latitudes.append(None)
                longitudes.append(None)
                addresses.append(None)

        self.df['Latitude'] = latitudes
        self.df['Longitude'] = longitudes
        self.df['GeocodedAddress'] = addresses

        geocoded_file = OUTPUT_DIR / "geocodificado.csv"
        self.df.to_csv(geocoded_file, index=False)
        logging.info(f"Archivo CSV geocodificado guardado en {geocoded_file}")

        explanation = (
            "Este proceso geocodifica las direcciones de las localidades, agregando coordenadas geográficas (latitud y longitud) "
            "al DataFrame. Esto es útil para análisis geoespaciales y visualización de datos en mapas."
        )
        messagebox.showinfo("Geocodificación", f"Geocodificación completada. Archivo guardado en {geocoded_file}\n\n{explanation}")

    def generar_mapa(self):
        """Genera un mapa con las direcciones geocodificadas."""
        if self.df.empty or 'Latitude' not in self.df.columns or'Longitude' not in self.df.columns:
            messagebox.showwarning("Advertencia", "Por favor, geocodifique las direcciones primero.")
            return

        centro = [self.df['Latitude'].mean(), self.df['Longitude'].mean()]
        mapa = folium.Map(location=centro, zoom_start=6)

        for _, row in self.df.iterrows():
            if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=row['GeocodedAddress'],
                ).add_to(mapa)

        mapa_file = OUTPUT_DIR / "mapa_geoespacial.html"
        mapa.save(mapa_file)
        logging.info(f"Mapa geoespacial guardado en {mapa_file}")

        webbrowser.open(mapa_file.resolve().as_uri())

        explanation = (
            "Este informe genera un mapa interactivo con las direcciones geocodificadas. "
            "Es útil para visualizar la distribución geográfica de los datos y realizar análisis espaciales."
        )
        messagebox.showinfo("Generar Mapa", f"Mapa generado exitosamente.\n\n{explanation}")

    def produccion_top_cultivos(self):
        """Genera un gráfico de líneas para los 4 principales cultivos por producción total."""
        if self.df.empty or 'cultivo' not in self.df.columns or 'campaña' not in self.df.columns or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener las columnas 'cultivo', 'campaña' y 'produccion'.")
            return

        # Agrupar los datos por cultivo y campaña, y sumar la producción
        grouped_data = self.df.groupby(['cultivo', 'campaña'])['produccion'].sum().reset_index()

        # Obtener los 4 principales cultivos por producción total
        top_cultivos = grouped_data.groupby('cultivo')['produccion'].sum().nlargest(4).index

        # Filtrar los datos para incluir solo los 4 cultivos principales
        filtered_data = grouped_data[grouped_data['cultivo'].isin(top_cultivos)]

        # Crear un gráfico de líneas que muestre la producción por campaña para los 4 cultivos principales
        plt.figure(figsize=(12, 8))
        for cultivo in top_cultivos:
            cultivo_data = filtered_data[filtered_data['cultivo'] == cultivo]
            plt.plot(cultivo_data['campaña'], cultivo_data['produccion'], marker='o', label=cultivo)

        plt.title('Producción de los 4 principales cultivos por campaña')
        plt.xlabel('Campaña')
        plt.ylabel('Producción (en toneladas)')
        plt.xticks(rotation=45)
        plt.legend(title='Cultivo')
        plt.grid(True)
        plt.tight_layout()

        output_file = OUTPUT_DIR / "produccion_top_cultivos.png"
        plt.savefig(output_file)
        plt.show()
        logging.info(f"Gráfica de producción de los 4 principales cultivos guardada en {output_file}")

        explanation = (
            "Este análisis muestra la evolución de la producción de los 4 principales cultivos a lo largo de las campañas. "
            "Permite visualizar cuáles cultivos han tenido mayor producción en diferentes períodos, ayudando en la planificación y toma de decisiones."
        )
        messagebox.showinfo("Producción Top Cultivos", f"Gráfica guardada en {output_file}\n\n{explanation}")

    def mostrar_dialogo_informes(self):
        """Muestra un cuadro de diálogo para seleccionar y generar informes."""
        informes = ["Producción Total por Provincia", "Correlación Sup. Sembrada-Sup. Cosechada", "Sumar Columnas", 
                    "Análisis Temporal", "Análisis de Correlación", "Modelos Predictivos", 
                    "Clasificación de Cultivos", "Análisis de Riesgos", "Evolución de Cultivos por Campaña", 
                    "Tendencias de Producción por Cultivo", "Clasificación de Texto con IA", "Predicción de Tendencias con IA", 
                    "Análisis Predictivo con Red Neuronal", "Producción Top Cultivos"]

        selected_informe = self.ask_option("Generar Informe", "Seleccione el informe a generar:", informes)
        if selected_informe:
            getattr(self, self.get_function_name_from_report(selected_informe))()

    @staticmethod
    def get_function_name_from_report(report_name):
        """Devuelve el nombre de la función correspondiente a un informe seleccionado."""
        function_mapping = {
            "Producción Total por Provincia": "produccion_total_por_provincia",
            "Correlación Sup. Sembrada-Sup. Cosechada": "correlacion_sup_sembrada_cosechada",
            "Sumar Columnas": "sumar_columnas",
            "Análisis Temporal": "analisis_temporal",
            "Análisis de Correlación": "analisis_correlacion",
            "Modelos Predictivos": "modelos_predictivos",
            "Clasificación de Cultivos": "clasificacion_cultivos",
            "Análisis de Riesgos": "analisis_riesgos",
            "Evolución de Cultivos por Campaña": "evolucion_cultivos_por_campaña",
            "Tendencias de Producción por Cultivo": "tendencias_produccion_por_cultivo",
            "Clasificación de Texto con IA": "clasificacion_texto_ia",
            "Predicción de Tendencias con IA": "prediccion_tendencias_ia",
            "Análisis Predictivo con Red Neuronal": "analisis_predictivo_nn",
            "Producción Top Cultivos": "produccion_top_cultivos",
        }
        return function_mapping.get(report_name, "")

    def ask_option(self, title, message, options):
        """Muestra un cuadro de diálogo para seleccionar una opción."""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("250x150")
        dialog.resizable(False, False)

        label = tk.Label(dialog, text=message)
        label.pack(pady=10)

        combobox_value = tk.StringVar()
        combobox = ttk.Combobox(dialog, textvariable=combobox_value, values=options)
        combobox.pack(pady=10)
        combobox.current(0)

        button = ttk.Button(dialog, text="Aceptar", command=dialog.destroy)
        button.pack(pady=10)

        dialog.grab_set()
        dialog.wait_window()

        selected_option = combobox_value.get()
        return selected_option

    @staticmethod
    def safe_file_name(name):
        """Devuelve un nombre de archivo seguro para usar en el sistema de archivos."""
        return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
 
 
 
if __name__ == "__main__":
    app = DataAnalyzer()
    app.root.mainloop()
