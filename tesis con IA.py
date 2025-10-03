import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from time import sleep
from geopy.geocoders import Nominatim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
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

# Configuración del registro de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directorio de salida para resultados
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Geolocalizador
geolocator = Nominatim(user_agent="geoapiExercises")


class FileHandler:
    """Manejo de archivos CSV."""

    @staticmethod
    def cargar_csv():
        """Carga un archivo CSV y devuelve un DataFrame."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()
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
    """Preprocesamiento y normalización de datos."""

    @staticmethod
    def normalize_text(text):
        """Elimina caracteres especiales y acentos."""
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
        return re.sub(r'[^\w\s]', '', text).lower().strip()

    @staticmethod
    def denormalize_text(normalized_text, original_texts):
        """Busca la versión original del texto normalizado."""
        for text in original_texts:
            if DataPreprocessing.normalize_text(text) == normalized_text:
                return text
        return None


class Visualization:
    """Visualización de datos."""

    @staticmethod
    def plot_bar_chart(data, title, xlabel, ylabel, output_file):
        """Genera y guarda una gráfica de barras."""
        if data.empty:
            logging.error("No hay datos disponibles para generar la gráfica.")
            return
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


class DataAnalyzer(tk.Tk):
    """Clase principal que maneja la interfaz y el análisis de datos."""

    def __init__(self):
        super().__init__()
        self.title("Aplicación de Análisis de Datos")
        self.geometry("600x400")
        self.df = pd.DataFrame()
        self.create_menu()

    def create_menu(self):
        """Crea el menú principal de la aplicación."""
        menu_bar = tk.Menu(self)
        self.config(menu=menu_bar)

        # Menú de archivo
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Cargar CSV", command=self.cargar_csv)
        file_menu.add_command(label="Salir", command=self.quit)
        menu_bar.add_cascade(label="Archivo", menu=file_menu)

        # Menú de análisis
        analysis_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Análisis", menu=analysis_menu)
        analysis_menu.add_command(label="Sumar Columnas", command=self.sumar_columnas)
        analysis_menu.add_command(label="Análisis Temporal", command=self.analisis_temporal)
        analysis_menu.add_command(label="Análisis de Correlación", command=self.analisis_correlacion)
        analysis_menu.add_command(label="Modelos Predictivos", command=self.modelos_predictivos)
        analysis_menu.add_command(label="Clasificación de Cultivos", command=self.clasificacion_cultivos)
        analysis_menu.add_command(label="Análisis de Riesgos", command=self.analisis_riesgos)
        analysis_menu.add_command(label="Producción Total por Provincia", command=self.produccion_total_por_provincia)
        analysis_menu.add_command(label="Evolución de Cultivos por Campaña", command=self.evolucion_cultivos_por_campaña)
        analysis_menu.add_command(label="Tendencias de Producción por Cultivo", command=self.tendencias_produccion_por_cultivo)
        analysis_menu.add_command(label="Clasificación de Texto con IA", command=self.clasificacion_texto_ia)
        analysis_menu.add_command(label="Predicción de Tendencias con IA", command=self.prediccion_tendencias_ia)
        analysis_menu.add_command(label="Análisis Predictivo con Red Neuronal", command=self.analisis_predictivo_nn)

        # Menú de geocodificación
        geocodificacion_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Geocodificación", menu=geocodificacion_menu)
        geocodificacion_menu.add_command(label="Geocodificar Direcciones", command=self.geocodificar_direcciones)
        geocodificacion_menu.add_command(label="Generar Mapa", command=self.generar_mapa)

    def cargar_csv(self):
        """Carga un archivo CSV utilizando la clase FileHandler."""
        self.df = FileHandler.cargar_csv()

    def sumar_columnas(self):
        """Genera una gráfica con la suma de columnas numéricas."""
        if self.df.empty:
            messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return
        suma_columnas = self.df.select_dtypes(include=[float, int]).sum()
        output_file = OUTPUT_DIR / "suma_columnas.png"
        Visualization.plot_bar_chart(suma_columnas, "Suma de Columnas Numéricas", "Columnas", "Suma", output_file)

    def analisis_temporal(self):
        """Genera un análisis temporal de la producción basado en la columna 'campaña'."""
        if self.df.empty or 'campaña' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener la columna 'campaña'.")
            return

        if 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener la columna 'produccion'.")
            return

        try:
            self.df['campaña'] = pd.to_numeric(self.df['campaña'], errors='coerce')
            summary_by_campaign = self.df.groupby('campaña').agg({
                'sup_sembrada': 'sum',
                'sup_cosechada': 'sum',
                'produccion': 'sum',
                'rendimiento': 'mean'
            }).reset_index()

            plt.figure(figsize=(14, 10))
            plt.plot(summary_by_campaign['campaña'], summary_by_campaign['produccion'], label='Producción', color='green')
            plt.title('Evolución de la Producción')
            plt.xlabel('Año de Campaña')
            plt.ylabel('Producción (toneladas)')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error en el análisis temporal: {e}")
            messagebox.showerror("Error", f"Error en el análisis temporal: {e}")

    def analisis_correlacion(self):
        """Genera una matriz de correlación entre las columnas numéricas."""
        if self.df.empty or self.df.select_dtypes(include=[float, int]).empty:
            messagebox.showwarning("Advertencia", "No hay columnas numéricas para analizar.")
            return

        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df.select_dtypes(include=[float, int]).corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Matriz de Correlación")
        plt.tight_layout()

        output_file = OUTPUT_DIR / "matriz_correlacion.png"
        plt.savefig(output_file)
        plt.show()
        logging.info(f"Matriz de correlación guardada en {output_file}")

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

        messagebox.showinfo("Modelo Predictivo", f"MSE: {mse:.2f}, R²: {r2:.2f}")

    def clasificacion_cultivos(self):
        """Clasifica cultivos utilizando RandomForestClassifier."""
        if self.df.empty or 'sup_sembrada' not in self.df.columns or 'cultivo' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener las columnas 'sup_sembrada' y 'cultivo'.")
            return

        X = self.df[['sup_sembrada']].values
        y = self.df['cultivo'].values

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)

        accuracy = classifier.score(X_test, y_test)
        messagebox.showinfo("Clasificación de Cultivos", f"Precisión del modelo: {accuracy:.2f}")

    def analisis_riesgos(self):
        """Análisis de riesgos utilizando clustering y PCA."""
        if self.df.empty or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener la columna 'produccion'.")
            return

        df_valid = self.df[['produccion']].dropna()
        scaler = StandardScaler()
        df_normalizado = scaler.fit_transform(df_valid)

        pca = PCA(n_components=2)
        df_reducido = pca.fit_transform(df_normalizado)

        dbscan = PCA(n_components=2)
        clusters = dbscan.fit(df_reducido)

        self.df['Cluster'] = clusters.labels_

        plt.figure(figsize=(10, 8))
        plt.scatter(df_reducido[:, 0], df_reducido[:, 1], c=clusters.labels_, cmap='viridis')
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.title("Clustering de Producción con DBSCAN")
        plt.colorbar(label='Cluster')

        clustering_file = OUTPUT_DIR / "clustering_produccion_dbscan.png"
        plt.savefig(clustering_file)
        plt.show()
        logging.info(f"Gráfica guardada en {clustering_file}")

    def produccion_total_por_provincia(self):
        """Genera un gráfico de la producción total por provincia."""
        if self.df.empty or 'provincia' not in self.df.columns or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener las columnas 'provincia' y 'produccion'.")
            return

        produccion_por_provincia = self.df.groupby('provincia')['produccion'].sum()
        output_file = OUTPUT_DIR / "produccion_por_provincia.png"
        Visualization.plot_bar_chart(produccion_por_provincia, "Producción Total por Provincia", "Provincia", "Producción", output_file)

    def evolucion_cultivos_por_campaña(self):
        """Genera un gráfico de la evolución de los cultivos por campaña."""
        if self.df.empty or 'campaña' not in self.df.columns or 'cultivo' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener las columnas 'campaña' y 'cultivo'.")
            return

        self.df['cultivo'] = self.df['cultivo'].apply(DataPreprocessing.normalize_text)
        cultivo_seleccionado = self.ask_option("Seleccione el cultivo", "Seleccionar Cultivo", self.df['cultivo'].unique())

        df_filtrado = self.df[self.df['cultivo'] == cultivo_seleccionado]
        columnas_presentes = ['sup_sembrada', 'sup_cosechada', 'produccion']
        columnas_presentes = [col for col in columnas_presentes if col in self.df.columns]

        if not columnas_presentes:
            messagebox.showwarning("Advertencia", "No hay datos para las columnas seleccionadas.")
            return

        plt.figure(figsize=(12, 8))
        for columna in columnas_presentes:
            df_filtrado.groupby('campaña')[columna].sum().plot(label=columna)

        plt.title(f"Evolución del Cultivo {cultivo_seleccionado} por Campaña")
        plt.xlabel("Campaña")
        plt.ylabel("Cantidad")
        plt.legend()
        plt.tight_layout()

        evolucion_file = OUTPUT_DIR / f"evolucion_cultivo_{cultivo_seleccionado}.png"
        plt.savefig(evolucion_file)
        plt.show()

    def tendencias_produccion_por_cultivo(self):
        """Genera un gráfico de tendencias de producción por cultivo y campaña."""
        if self.df.empty or 'campaña' not in self.df.columns or 'cultivo' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener las columnas 'campaña', 'cultivo' y 'produccion'.")
            return

        plt.figure(figsize=(12, 8))
        for cultivo in self.df['cultivo'].unique():
            subset = self.df[self.df['cultivo'] == cultivo]
            plt.plot(subset['campaña'], subset['produccion'], label=cultivo)

        plt.title("Tendencias de Producción por Cultivo y Campaña")
        plt.xlabel("Campaña")
        plt.ylabel("Producción")
        plt.legend(title="Cultivo")
        plt.tight_layout()

        tendencias_file = OUTPUT_DIR / "tendencias_produccion.png"
        plt.savefig(tendencias_file)
        plt.show()

    def clasificacion_texto_ia(self):
        """Clasifica textos utilizando un modelo de IA."""
        if self.df.empty or 'texto' not in self.df.columns or 'categoria' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener las columnas 'texto' y 'categoria'.")
            return

        X = self.df['texto'].apply(DataPreprocessing.normalize_text).values
        y = self.df['categoria'].values

        vectorizer = TfidfVectorizer()
        X_vectorized = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

        classifier = MultinomialNB()
        classifier.fit(X_train, y_train)

        accuracy = classifier.score(X_test, y_test)
        messagebox.showinfo("Clasificación de Texto con IA", f"Precisión del modelo: {accuracy:.2f}")

    def prediccion_tendencias_ia(self):
        """Predice tendencias de producción utilizando un modelo avanzado de SVR."""
        if self.df.empty or 'año' not in self.df.columns or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener las columnas 'año' y 'produccion'.")
            return

        X = self.df[['año']].values
        y = self.df['produccion'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        messagebox.showinfo("Predicción de Tendencias con IA", f"MSE: {mse:.2f}, R²: {r2:.2f}")

    def analisis_predictivo_nn(self):
        """Realiza análisis predictivo utilizando redes neuronales."""
        if self.df.empty or 'sup_sembrada' not in self.df.columns or 'sup_cosechada' not in self.df.columns or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener las columnas 'sup_sembrada', 'sup_cosechada' y 'produccion'.")
            return

        features = self.df[['sup_sembrada', 'sup_cosechada']].values
        target = self.df['produccion'].values

        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        target_scaled = scaler.fit_transform(target.reshape(-1, 1)).ravel()

        X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.2, random_state=42)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=100, validation_split=0.2)

        loss = model.evaluate(X_test, y_test)
        messagebox.showinfo("Análisis Predictivo con NN", f"Pérdida en el conjunto de prueba: {loss}")

    def geocodificar_direcciones(self):
        """Geocodifica direcciones basadas en 'departamento', 'provincia', y 'pais'."""
        if self.df.empty or 'departamento' not in self.df.columns or 'provincia' not in self.df.columns or 'pais' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener las columnas 'departamento', 'provincia' y 'pais'.")
            return

        def geocode_with_retry(address):
            try:
                location = geolocator.geocode(address)
                return location
            except (GeocoderTimedOut, GeocoderServiceError):
                sleep(1)
                return geocode_with_retry(address)

        latitudes, longitudes, addresses = [], [], []
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
        messagebox.showinfo("Geocodificación", f"Archivo CSV guardado en {geocoded_file}")

    def generar_mapa(self):
        """Genera un mapa interactivo basado en coordenadas geocodificadas."""
        if self.df.empty or 'Latitude' not in self.df.columns or 'Longitude' not in self.df.columns:
            messagebox.showwarning("Advertencia", "Primero geocodifique las direcciones.")
            return

        centro = [self.df['Latitude'].mean(), self.df['Longitude'].mean()]
        mapa = folium.Map(location=centro, zoom_start=6)

        for _, row in self.df.iterrows():
            if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=row['GeocodedAddress']
                ).add_to(mapa)

        mapa_file = OUTPUT_DIR / "mapa_geoespacial.html"
        mapa.save(mapa_file)
        webbrowser.open(mapa_file.resolve().as_uri())
        messagebox.showinfo("Generar Mapa", f"Mapa guardado en {mapa_file}")

    def ask_option(self, title, message, options):
        """Muestra un cuadro de diálogo para seleccionar una opción."""
        dialog = tk.Toplevel(self)
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

        return combobox_value.get()


if __name__ == "__main__":
    app = DataAnalyzer()
    app.mainloop()
