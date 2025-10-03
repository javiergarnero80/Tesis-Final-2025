import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
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
from sklearn.ensemble import RandomForestRegressor
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
import requests
import os
from PIL import Image, ImageTk

# Logging configuration
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Directorio de salida
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Carpeta específica para figuras de la tesis
FIGS_TESIS_DIR = OUTPUT_DIR / "figs_tesis"
FIGS_TESIS_DIR.mkdir(parents=True, exist_ok=True)

def human_readable_magnitude(value, _):
    """Formatea valores numéricos con sufijos (K, M, B) para ejes académicos."""
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"

def bootstrap_mean_ci(series, n_boot=2000, ci=95, rng=None):
    """
    Calcula un intervalo de confianza bootstrap para la media de una serie.

    Parameters
    ----------
    series : pandas.Series
        Serie numérica sin valores NaN para replicar.
    n_boot : int, optional
        Cantidad de réplicas bootstrap a generar. Por defecto 2000.
    ci : int, optional
        Nivel de confianza (en porcentaje). Por defecto 95.
    rng : numpy.random.Generator, optional
        Generador de números aleatorios para reproducibilidad.

    Returns
    -------
    tuple[float, float]
        Valores (lower, upper) correspondientes al intervalo de confianza.
    """
    clean_data = series.dropna().to_numpy()
    if clean_data.size == 0:
        return (np.nan, np.nan)

    rng = rng or np.random.default_rng(42)
    resamples = rng.choice(clean_data, size=(n_boot, clean_data.size), replace=True)
    boot_means = resamples.mean(axis=1)
    alpha = (100 - ci) / 2
    lower = np.percentile(boot_means, alpha)
    upper = np.percentile(boot_means, 100 - alpha)
    return (lower, upper)

def sumar_columnas(df, cols):
    """Genera reportes visuales y opcionalmente guarda gráficos estadísticos para columnas numéricas.

    Parámetros:
        df (pandas.DataFrame): DataFrame con los datos a analizar.
        cols (list): Lista de columnas numéricas a analizar.

    Descripción:
        Esta función muestra 4 gráficos académicos sobre los datos numéricos en una figura con 4 subplots:
        1. Fig01_totales.png: Barras de totales acumulados (ordenadas de mayor a menor).
        2. Fig02_promedios.png: Barras de promedios con intervalos de confianza al 95%.
        3. Fig03_cv.png: Coeficiente de variación por variable.
        4. Fig04_min_prom_max.png: Comparación Min-Promedio-Máximo.

        Los gráficos se muestran en pantalla en una sola ventana. Después, se pide confirmación en consola para guardar.
        Si se responde 's', se solicita carpeta de destino y se guardan a 300 dpi.
        Si 'n', no se guarda nada.
        Estilo gráfico académico: barras ordenadas mayor a menor, ejes con K/M/B, títulos y etiquetas claras, colores sobrios (azul, verde, gris).

    Notas:
        - Utiliza bootstrap para calcular intervalos de confianza del 95%.
        - Maneja errores de entrada y proporciona validaciones robustas.
        - Adecuado para tesis académicas y análisis profesional.
    """
    # Validar la entrada y quedarnos solo con las columnas numéricas disponibles en el DataFrame.
    if df is None or df.empty:
        raise ValueError("El DataFrame no contiene información para analizar.")

    numeric_cols = [col for col in cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_cols:
        raise ValueError("La lista de columnas no contiene variables numéricas válidas.")

    data = df[numeric_cols].dropna()
    if data.empty:
        raise ValueError("No hay registros completos disponibles para las columnas indicadas.")

    # Preparar estilo y orden de las variables por importancia (suma descendente).
    sns.set_theme(style="whitegrid", context="notebook")  # Mejor para tesis
    plt.rcParams['font.family'] = 'serif'  # Fuente serif académica
    plt.rcParams['font.size'] = 10  # Tamaño de fuente consistente
    totals = data.sum().sort_values(ascending=False)
    ordered_cols = totals.index.tolist()

    # Calcular estadísticos claves alineados al orden definido.
    means = data[ordered_cols].mean()
    stds = data[ordered_cols].std()
    mean_safe = means.replace(0, np.nan)  # Evita divisiones por cero al calcular el CV.
    cv = (stds / mean_safe) * 100
    mins = data[ordered_cols].min()
    maxs = data[ordered_cols].max()

    # Configurar formato de etiquetas del eje X.
    def _format_xticklabels(axis):
        for label in axis.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            label.set_fontsize(9)

    # Calcular intervalos de confianza bootstrap para cada columna.
    lower_errors = []
    upper_errors = []
    for col in ordered_cols:
        lower_ci, upper_ci = bootstrap_mean_ci(data[col])
        lower_errors.append(max(0, means[col] - lower_ci) if not np.isnan(lower_ci) else 0)
        upper_errors.append(max(0, upper_ci - means[col]) if not np.isnan(upper_ci) else 0)

    # Configurar matplotlib para modo interactivo y evitar bloqueo de tkinter
    plt.ion()  # Activar modo interactivo
    plt.show(block=False)  # No bloquear

    # Crear figura única con 4 subplots para mostrar todos los gráficos juntos.
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Análisis Estadístico Integral de Variables Numéricas", fontsize=18, fontweight='bold', y=0.98)

    # Subplot 1: Totales acumulados por columna (ordenados descendente).
    bars1 = ax1.bar(ordered_cols, totals.values, color="#4C72B0", edgecolor="black", alpha=0.9)
    ax1.set_title("a) Totales acumulados por variable", fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel("Variables numéricas", fontsize=11)
    ax1.set_ylabel("Totales", fontsize=11)
    ax1.yaxis.set_major_formatter(FuncFormatter(human_readable_magnitude))
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.margins(x=0.05)
    _format_xticklabels(ax1)
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(human_readable_magnitude(height, None),
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 7), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Subplot 2: Promedios con intervalo de confianza al 95%.
    bars2 = ax2.bar(ordered_cols, means.values, color="#55A868", edgecolor="black", alpha=0.9,
                    yerr=[lower_errors, upper_errors], capsize=4, ecolor="#2E8B57")
    ax2.set_title("b) Promedios con intervalo de confianza 95%", fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel("Variables numéricas", fontsize=11)
    ax2.set_ylabel("Promedios", fontsize=11)
    ax2.yaxis.set_major_formatter(FuncFormatter(human_readable_magnitude))
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.margins(x=0.05)
    _format_xticklabels(ax2)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax2.annotate(human_readable_magnitude(height, None),
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 7), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Subplot 3: Coeficiente de variación.
    bars3 = ax3.bar(ordered_cols, cv.values, color="#7F7F7F", edgecolor="black", alpha=0.9)
    ax3.set_title("c) Coeficiente de variación", fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel("Variables numéricas", fontsize=11)
    ax3.set_ylabel("CV (%)", fontsize=11)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    ax3.margins(x=0.05)
    _format_xticklabels(ax3)
    ax3.axhline(y=50, color="#4C72B0", linestyle="--", linewidth=2, alpha=0.8, label="Variabilidad moderada")
    ax3.axhline(y=100, color="#55A868", linestyle="--", linewidth=2, alpha=0.8, label="Alta variabilidad")
    ax3.legend(fontsize=9, loc='upper right')
    for bar in bars3:
        height = bar.get_height()
        ax3.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 7), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Subplot 4: Comparación Min-Promedio-Máximo con barras agrupadas.
    width = 0.25
    x_pos = np.arange(len(ordered_cols))
    ax4.bar(x_pos - width, mins.values, width, label="Mínimo", color="#4C72B0", edgecolor="black", alpha=0.9)
    ax4.bar(x_pos, means.values, width, label="Promedio", color="#55A868", edgecolor="black", alpha=0.9)
    ax4.bar(x_pos + width, maxs.values, width, label="Máximo", color="#7F7F7F", edgecolor="black", alpha=0.9)
    ax4.set_title("d) Comparación.min-prm-max", fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlabel("Variables numéricas", fontsize=11)
    ax4.set_ylabel("Valores", fontsize=11)
    ax4.yaxis.set_major_formatter(FuncFormatter(human_readable_magnitude))
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(ordered_cols, rotation=45, ha='right')
    ax4.grid(axis='y', linestyle='--', alpha=0.5)
    ax4.legend(fontsize=9, loc='upper left')
    ax4.margins(x=0.05)

    # Ajustar layout para evitar superposiciones.
    fig.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.4, wspace=0.25)

    # Mostrar la figura completa con los 4 gráficos.
    plt.show()

    # Consultar al usuario si desea guardar los gráficos.
    respuesta = input("¿Desea guardar los gráficos como PNG? (s/n): ").strip().lower()
    while respuesta not in {"s", "n"}:
        respuesta = input("Respuesta no válida. Ingrese 's' para sí o 'n' para no: ").strip().lower()

    if respuesta == "s":
        carpeta = input("Ingrese la carpeta donde desea guardar los gráficos: ").strip()
        while not carpeta:
            carpeta = input("La ruta no puede estar vacía. Ingrese la carpeta destino: ").strip()
        destino = Path(carpeta).expanduser()
        destino.mkdir(parents=True, exist_ok=True)
        fig.savefig(destino / "analisis_estadistico_integral.png", dpi=300, bbox_inches='tight')
    else:
        print("Los gráficos no se guardaron.")

    # Cerrar las figuras para liberar memoria en sesiones iterativas.
    plt.close(fig)


# Geolocalizador con configuración mejorada
geolocator = Nominatim(user_agent="analisis_agricola_app/1.0")

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
                # Normalizar nombres de columnas para manejar acentos y mayúsculas/minúsculas
                df.columns = df.columns.str.normalize('NFD').str.encode('ascii', 'ignore').str.decode('utf-8').str.lower()
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
    def plot_bar_chart(data, title, xlabel, ylabel, output_file, function_name=""):
        """Genera una gráfica de barras."""
        fig = plt.figure(figsize=(12, 8))
        if function_name:
            fig.suptitle(f"{function_name}", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
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
        self.root.title("🌾 Aplicación de Análisis Agrícola con IA")
        self.root.geometry("600x400")

        # Intentar cargar ícono
        self.cargar_icono()

        self.df = pd.DataFrame()
        self.setup_menu()

    def cargar_icono(self):
        """Carga un ícono para la aplicación si está disponible."""
        try:
            # Buscar ícono en el directorio actual
            icono_path = Path("icono_agricola.png")
            if icono_path.exists():
                # Cargar imagen con PIL
                imagen = Image.open(icono_path)
                # Redimensionar si es necesario
                imagen = imagen.resize((64, 64), Image.Resampling.LANCZOS)
                icono = ImageTk.PhotoImage(imagen)
                self.root.iconphoto(True, icono)
            else:
                # Crear ícono simple si no hay imagen
                self.crear_icono_simple()
        except Exception as e:
            logging.warning(f"No se pudo cargar ícono: {e}")
            # Crear ícono simple como fallback
            self.crear_icono_simple()

    def crear_icono_simple(self):
        """Crea un ícono simple agrícola usando canvas."""
        try:
            # Crear una imagen simple de 64x64 con colores agrícolas
            canvas = tk.Canvas(self.root, width=64, height=64, highlightthickness=0)
            canvas.pack()

            # Fondo verde (campo)
            canvas.create_rectangle(0, 0, 64, 64, fill='#4CAF50', outline='')

            # Tallos de trigo simplificados
            for i in range(3):
                x = 16 + i * 16
                # Tallo
                canvas.create_line(x, 45, x, 20, fill='#8BC34A', width=2)
                # Espigas
                canvas.create_oval(x-3, 15, x+3, 25, fill='#CDDC39', outline='')

            # Sol simplificado
            canvas.create_oval(45, 10, 55, 20, fill='#FFC107', outline='')

            # Convertir canvas a imagen
            from PIL import ImageGrab
            # Esta parte puede no funcionar en todos los sistemas
            # Como alternativa, usaremos un ícono por defecto si está disponible

            # Intentar usar un ícono por defecto del sistema
            try:
                self.root.iconbitmap(default='')  # Esto puede funcionar en algunos sistemas
            except:
                pass

            canvas.destroy()  # Limpiar canvas temporal

        except Exception as e:
            logging.warning(f"No se pudo crear ícono simple: {e}")
    def _check_csv_loaded(self):
        """Verifica si el CSV está cargado y muestra un mensaje si no lo está."""
        if self.df.empty:
            messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return False
        return True

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
        self.analisis_menu.add_command(label="Evolución de Cultivos por Campaña", command=self.evolucion_cultivos_por_campana)
        self.analisis_menu.add_command(label="Tendencias de Producción por Cultivo", command=self.tendencias_produccion_por_cultivo)
        self.analisis_menu.add_command(label="Predicción de Tendencias con IA", command=self.prediccion_tendencias_ia)
        self.analisis_menu.add_command(label="Análisis Predictivo con Red Neuronal", command=self.analisis_predictivo_nn)
        self.analisis_menu.add_command(label="Producción Top Cultivos", command=self.produccion_top_cultivos)

        self.geocodificacion_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Geocodificación", menu=self.geocodificacion_menu)
        self.geocodificacion_menu.add_command(label="Geocodificar Direcciones", command=self.geocodificar_direcciones)
        self.geocodificacion_menu.add_command(label="Generar Mapa", command=self.generar_mapa)
        self.geocodificacion_menu.add_command(label="Mapa de Distribución de Cultivos", command=self.mapa_distribucion_cultivos)

        # Menú de Ayuda
        self.help_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Ayuda", menu=self.help_menu)
        self.help_menu.add_command(label="Acerca de", command=self.acerca_de)
        self.help_menu.add_command(label="Manual de Usuario", command=self.manual_usuario)

    def cargar_csv(self):
        """Carga un archivo CSV utilizando la clase FileHandler."""
        self.df = FileHandler.cargar_csv()

    def sumar_columnas(self):
        """Realiza un análisis estadístico integral de las variables numéricas del dataset agrícola."""
        if not self._check_csv_loaded():
            return

        # Obtener columnas numéricas, excluyendo coordenadas geocodificadas
        numeric_cols = [col for col in self.df.select_dtypes(include=[float, int]).columns
                       if col not in ['Latitude', 'Longitude']]

        if not numeric_cols:
            messagebox.showwarning("Advertencia", "No se encontraron columnas numéricas para analizar.")
            return

        # Filtrar datos válidos (sin NaN)
        df_numeric = self.df[numeric_cols].dropna()

        if len(df_numeric) < 10:
            messagebox.showwarning("Advertencia", "No hay suficientes datos válidos para el análisis estadístico.")
            return

        # Calcular estadísticas descriptivas completas
        estadisticas = df_numeric.describe()
        suma_columnas = df_numeric.sum()
        mediana_columnas = df_numeric.median()
        desviacion_columnas = df_numeric.std()
        coef_variacion = (desviacion_columnas / df_numeric.mean()) * 100

        # Identificar variables más importantes
        variable_mayor_suma = suma_columnas.idxmax()
        variable_mayor_variabilidad = coef_variacion.idxmax()
        variable_mas_estable = coef_variacion.idxmin()

        # Crear visualización mejorada con 3 subgráficos principales
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Gráfico 1: Totales por variable (suma)
        suma_columnas.plot(kind='bar', ax=ax1, color='lightblue', edgecolor='navy')
        ax1.set_title('a) Totales Acumulados por Variable')
        ax1.set_xlabel('Variables Numéricas')
        ax1.set_ylabel('Suma Total')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Agregar valores en las barras
        for i, v in enumerate(suma_columnas.values):
            ax1.text(i, v + v*0.01, f'{v:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Gráfico 2: Promedios por variable
        promedios = df_numeric.mean()
        promedios.plot(kind='bar', ax=ax2, color='lightgreen', edgecolor='darkgreen')
        ax2.set_title('b) Valores Promedio por Variable')
        ax2.set_xlabel('Variables Numéricas')
        ax2.set_ylabel('Promedio')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # Agregar valores en las barras
        for i, v in enumerate(promedios.values):
            ax2.text(i, v + v*0.01, f'{v:,.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Gráfico 3: Coeficiente de variación (estabilidad)
        colores_cv = ['red' if cv > 100 else 'orange' if cv > 50 else 'green' for cv in coef_variacion.values]
        coef_variacion.plot(kind='bar', ax=ax3, color=colores_cv, edgecolor='black')
        ax3.set_title('c) Coeficiente de Variación por Variable (%)')
        ax3.set_xlabel('Variables Numéricas')
        ax3.set_ylabel('Coeficiente de Variación (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Variabilidad Media (50%)')
        ax3.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Alta Variabilidad (100%)')
        ax3.legend()

        # Agregar valores en las barras
        for i, v in enumerate(coef_variacion.values):
            ax3.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

        plt.suptitle("sumar_columnas", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88, wspace=0.3)

        # Guardar gráfico
        output_file = OUTPUT_DIR / "analisis_estadistico_integral.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        logging.info(f"Análisis estadístico integral guardado en {output_file}")

        # Identificar outliers usando el método IQR para todas las variables
        outliers_info = []
        for col in numeric_cols:
            Q1 = df_numeric[col].quantile(0.25)
            Q3 = df_numeric[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df_numeric[(df_numeric[col] < lower_bound) | (df_numeric[col] > upper_bound)][col]
            if len(outliers) > 0:
                outliers_info.append(f"{col}: {len(outliers)} valores atípicos ({len(outliers)/len(df_numeric)*100:.1f}%)")

        # Crear reporte detallado
        outliers_texto = "\n".join(outliers_info[:5]) if outliers_info else "No se detectaron valores atípicos significativos"

        # Crear explicaciones detalladas para cada suma
        explicaciones_sumas = []
        for col in suma_columnas.index:
            col_lower = col.lower()
            suma_valor = suma_columnas[col]

            if 'sup' in col_lower and 'sembrada' in col_lower:
                explicacion = f"Superficie total sembrada: {suma_valor:,.0f} hectáreas. Representa la extensión total de tierra utilizada para siembra en el período analizado."
            elif 'sup' in col_lower and 'cosechada' in col_lower:
                explicacion = f"Superficie total cosechada: {suma_valor:,.0f} hectáreas. Área efectiva que se logró cosechar exitosamente."
            elif 'prod' in col_lower:
                explicacion = f"Producción total: {suma_valor:,.0f} toneladas. Volumen total de cultivos obtenidos en el período."
            elif 'rend' in col_lower:
                explicacion = f"Rendimiento total acumulado: {suma_valor:,.0f} kg/ha. Suma de todos los rendimientos individuales registrados."
            else:
                explicacion = f"Total acumulado de '{col}': {suma_valor:,.0f}. Suma de todos los valores registrados para esta variable."

            explicaciones_sumas.append(f"   • {explicacion}")

        explicaciones_sumas_texto = "\n".join(explicaciones_sumas)

        explanation = (
            "📊 ANÁLISIS ESTADÍSTICO DE VARIABLES NUMÉRICAS\n\n"
            "Este análisis examina todas las columnas numéricas de tus datos "
            "agrícolas mostrando totales, promedios y estabilidad de cada variable.\n\n"
            f"🔍 Lo que se analizó: {len(df_numeric):,} registros con datos completos\n"
            f"📈 Columnas numéricas encontradas: {len(numeric_cols)}\n\n"
            "📊 LOS 3 GRÁFICOS MUESTRAN:\n"
            "   a) Totales acumulados - Qué variables tienen los números más grandes\n"
            "   b) Valores promedio - El promedio de cada variable\n"
            "   c) Coeficiente de variación - Qué variables son más estables\n\n"
            "🏆 RESULTADOS PRINCIPALES:\n"
            f"   • Variable con mayor suma total: {variable_mayor_suma} "
            f"(total: {suma_columnas[variable_mayor_suma]:,.0f})\n"
            f"   • Variable más variable: {variable_mayor_variabilidad} (cambia mucho)\n"
            f"   • Variable más estable: {variable_mas_estable} (cambia poco)\n\n"
            "📊 NÚMEROS BÁSICOS:\n"
            f"   • Promedio general de todas las variables: {df_numeric.mean().mean():,.1f}\n"
            f"   • Valores atípicos detectados: {outliers_texto}\n\n"
            "💰 EXPLICACIÓN DETALLADA DE CADA SUMA:\n"
            f"{explicaciones_sumas_texto}\n\n"
            "💡 INTERPRETACIÓN:\n"
            "   • Barras más altas = variables más importantes\n"
            "   • Colores en gráfico c): Verde=estable, Naranja=medio, Rojo=variable\n"
            "   • Valores atípicos pueden indicar errores o casos especiales\n\n"
            "📋 UTILIDAD PRÁCTICA:\n"
            "   • Identificar variables más relevantes para análisis\n"
            "   • Detectar problemas de calidad en los datos\n"
            "   • Guiar decisiones sobre qué variables priorizar"
        )

        messagebox.showinfo("Análisis Estadístico Integral", f"Análisis completado y guardado en {output_file}\n\n{explanation}")

    def analisis_temporal(self):
        """Genera un análisis temporal de la producción."""
        if not self._check_csv_loaded():
            return
        if 'campana' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener la columna 'campana'.")
            return

        if 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener la columna 'produccion'.")
            return

        # Integración del nuevo análisis temporal
        self.df['campana'] = self.df['campana'].astype(str).str.split('/').str[0].astype(int)
        summary_by_campaign = self.df.groupby('campana').agg({
            'sup_sembrada': 'sum',
            'sup_cosechada': 'sum',
            'produccion': 'sum',
            'rendimiento': 'mean'
        }).reset_index()
        summary_by_campaign.sort_values(by='campana', inplace=True)

        plt.figure(figsize=(14, 10))

        # Superficie Sembrada y Cosechada
        plt.subplot(2, 2, 1)
        plt.plot(summary_by_campaign['campana'], summary_by_campaign['sup_sembrada'], label='Superficie Sembrada')
        plt.plot(summary_by_campaign['campana'], summary_by_campaign['sup_cosechada'], label='Superficie Cosechada')
        plt.title('Evolución de la Superficie Sembrada y Cosechada')
        plt.xlabel('Año de Campaña')
        plt.ylabel('Superficie (hectáreas)')
        plt.legend()

        # Producción
        plt.subplot(2, 2, 2)
        plt.plot(summary_by_campaign['campana'], summary_by_campaign['produccion'], label='Producción', color='green')
        plt.title('Evolución de la Producción')
        plt.xlabel('Año de Campaña')
        plt.ylabel('Producción (toneladas)')

        # Rendimiento
        plt.subplot(2, 2, 3)
        plt.plot(summary_by_campaign['campana'], summary_by_campaign['rendimiento'], label='Rendimiento', color='orange')
        plt.title('Evolución del Rendimiento Promedio')
        plt.xlabel('Año de Campaña')
        plt.ylabel('Rendimiento (kg/ha)')

        plt.suptitle("analisis_temporal", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # Análisis adicional de tendencias temporales
        años_analizados = len(summary_by_campaign)
        año_inicial = summary_by_campaign['campana'].min()
        año_final = summary_by_campaign['campana'].max()

        # Calcular tendencias para cada variable
        tendencias = {}
        for columna in ['sup_sembrada', 'sup_cosechada', 'produccion', 'rendimiento']:
            if len(summary_by_campaign) > 1:
                valor_inicial = summary_by_campaign[columna].iloc[0]
                valor_final = summary_by_campaign[columna].iloc[-1]
                if valor_inicial > 0:
                    cambio_porcentual = ((valor_final - valor_inicial) / valor_inicial) * 100
                    tendencias[columna] = cambio_porcentual
                else:
                    tendencias[columna] = 0

        # Identificar años con valores extremos
        max_produccion_año = summary_by_campaign.loc[summary_by_campaign['produccion'].idxmax(), 'campana']
        min_produccion_año = summary_by_campaign.loc[summary_by_campaign['produccion'].idxmin(), 'campana']

        # Calcular eficiencia (relación entre superficie cosechada y sembrada)
        summary_by_campaign['eficiencia_cosecha'] = (summary_by_campaign['sup_cosechada'] / summary_by_campaign['sup_sembrada']) * 100
        eficiencia_promedio = summary_by_campaign['eficiencia_cosecha'].mean()

        explanation = (
            f"📈 ANÁLISIS TEMPORAL DE PRODUCCIÓN AGRÍCOLA\n\n"
            f"📅 Período analizado: {año_inicial} - {año_final} ({años_analizados} campañas)\n\n"
            f"📊 TENDENCIAS OBSERVADAS:\n"
            f"   • Superficie sembrada: {tendencias.get('sup_sembrada', 0):+.1f}% cambio total\n"
            f"   • Superficie cosechada: {tendencias.get('sup_cosechada', 0):+.1f}% cambio total\n"
            f"   • Producción total: {tendencias.get('produccion', 0):+.1f}% cambio total\n"
            f"   • Rendimiento promedio: {tendencias.get('rendimiento', 0):+.1f}% cambio total\n\n"
            f"🏆 AÑOS DESTACADOS:\n"
            f"   • Mayor producción: Campaña {max_produccion_año}\n"
            f"   • Menor producción: Campaña {min_produccion_año}\n\n"
            f"⚡ EFICIENCIA AGRÍCOLA:\n"
            f"   • Eficiencia promedio de cosecha: {eficiencia_promedio:.1f}%\n"
            f"     ↳ Porcentaje de superficie sembrada que se cosecha exitosamente\n\n"
            f"💡 INSIGHTS CLAVE:\n"
            f"   • {'Tendencia positiva' if tendencias.get('produccion', 0) > 5 else 'Tendencia negativa' if tendencias.get('produccion', 0) < -5 else 'Producción estable'} en producción total\n"
            f"   • {'Mejora' if tendencias.get('rendimiento', 0) > 0 else 'Declive'} en eficiencia productiva\n"
            f"   • Eficiencia de cosecha {'alta (>85%)' if eficiencia_promedio > 85 else 'media (70-85%)' if eficiencia_promedio > 70 else 'baja (<70%)'}\n\n"
            f"📋 RECOMENDACIONES:\n"
            f"   • Analizar causas de variaciones extremas en años de {max_produccion_año} y {min_produccion_año}\n"
            f"   • {'Mejorar' if eficiencia_promedio < 80 else 'Mantener'} prácticas de cosecha\n"
            f"   • Monitorear tendencias de rendimiento para optimizar cultivos"
        )

        messagebox.showinfo("Análisis Temporal", explanation)

    def analisis_correlacion(self):
        """Genera análisis de correlación con diseño profesional y limpio."""
        if not self._check_csv_loaded():
            return

        numeric_df = self.df.select_dtypes(include=[float, int])
        if numeric_df.empty:
            messagebox.showwarning("Advertencia", "No hay columnas numéricas para analizar.")
            return

        # Calcular matriz de correlación
        correlation_matrix = numeric_df.corr()

        # Crear figura principal con diseño profesional
        fig = plt.figure(figsize=(18, 12), facecolor='white')
        fig.suptitle('ANÁLISIS DE CORRELACIÓN AGRÍCOLA',
                    fontsize=24, fontweight='bold', y=0.95,
                    color='#2C3E50', family='Arial')

        # ==========================================
        # GRÁFICO 1: DICCIONARIO LIMPIO Y PROFESIONAL
        # ==========================================
        ax1 = plt.subplot(2, 2, 1)
        ax1.set_facecolor('#F8F9FA')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('#CCCCCC')
        ax1.spines['bottom'].set_color('#CCCCCC')

        ax1.text(0.5, 0.95, 'GUÍA DE INTERPRETACIÓN',
                ha='center', va='top', fontsize=16, fontweight='bold', color='#2C3E50')
        ax1.text(0.5, 0.85, 'Aprende a interpretar los coeficientes de correlación',
                ha='center', va='top', fontsize=10, color='#7F8C8D')

        # Crear diccionario más limpio y profesional
        correlations_info = [
            ("CORRELACIÓN POSITIVA FUERTE", "+0.7 a +1.0",
             "Las variables aumentan juntas", "#27AE60"),
            ("CORRELACIÓN NEGATIVA FUERTE", "-1.0 a -0.7",
             "Cuando una sube, la otra baja", "#E74C3C"),
            ("CORRELACIÓN MODERADA", "±0.3 a ±0.7",
             "Relación moderada entre variables", "#F39C12"),
            ("CORRELACIÓN DÉBIL", "-0.3 a +0.3",
             "Las variables actúan independientemente", "#95A5A6")
        ]

        y_pos = 0.65
        for name, range_val, description, color in correlations_info:
            # Título de la correlación
            ax1.text(0.05, y_pos, name, fontsize=11, fontweight='bold', color=color)
            ax1.text(0.55, y_pos, range_val, fontsize=10, color='#2C3E50')

            # Descripción
            ax1.text(0.05, y_pos - 0.08, description, fontsize=9, color='#34495E')

            # Ejemplo agrícola
            ax1.text(0.05, y_pos - 0.15, "Ejemplo agrícola:", fontsize=9, fontweight='bold', color='#2C3E50')

            y_pos -= 0.25

        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')

        # ==========================================
        # GRÁFICO 2: MATRIZ DE CORRELACIÓN PROFESIONAL
        # ==========================================
        ax2 = plt.subplot(2, 2, 2)

        # Seleccionar variables más importantes
        important_vars = []
        for col in correlation_matrix.columns:
            if any(keyword in col.lower() for keyword in ['sup', 'prod', 'rend', 'camp']):
                important_vars.append(col)

        if len(important_vars) >= 2:
            subset_corr = correlation_matrix.loc[important_vars, important_vars]
            matrix_data = subset_corr
            title = 'Variables Principales Agrícolas'
        else:
            matrix_data = correlation_matrix
            title = 'Todas las Variables'

        # Crear heatmap más profesional
        mask = np.triu(np.ones_like(matrix_data, dtype=bool))
        sns.heatmap(matrix_data, mask=mask, annot=True, cmap='RdYlBu_r', fmt='.2f',
                   square=True, ax=ax2, cbar_kws={'shrink': 0.8, 'aspect': 20},
                   annot_kws={'fontsize': 10, 'fontweight': 'bold'},
                   linewidths=0.5, linecolor='white')

        ax2.set_title(f'MATRIZ DE CORRELACIÓN\n{title}', fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
        ax2.tick_params(axis='x', rotation=45, labelsize=9)
        ax2.tick_params(axis='y', labelsize=9)

        # ==========================================
        # GRÁFICO 3: TOP RELACIONES MÁS IMPORTANTES
        # ==========================================
        ax3 = plt.subplot(2, 2, 3)
        ax3.set_facecolor('#F8F9FA')

        all_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                var1 = correlation_matrix.columns[i]
                var2 = correlation_matrix.columns[j]
                all_correlations.append((var1, var2, corr_val))

        # Ordenar por valor absoluto
        all_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        top_5 = all_correlations[:5]

        if top_5:
            # Crear etiquetas limpias
            labels = []
            for pair in top_5:
                var1_clean = pair[0].replace('_', ' ').title()
                var2_clean = pair[1].replace('_', ' ').title()
                labels.append(f"{var1_clean}\nvs\n{var2_clean}")

            values = [pair[2] for pair in top_5]

            # Colores profesionales
            colors = ['#27AE60' if v > 0.3 else '#E74C3C' if v < -0.3 else '#F39C12' for v in values]

            bars = ax3.bar(range(len(labels)), values, color=colors, alpha=0.8,
                          edgecolor='white', linewidth=1)

            ax3.set_xticks(range(len(labels)))
            ax3.set_xticklabels(labels, rotation=0, ha='center', fontsize=8, fontweight='bold')
            ax3.set_ylabel('Coeficiente de Correlación', fontsize=11, color='#2C3E50')
            ax3.set_title('RELACIONES MÁS IMPORTANTES', fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
            ax3.grid(True, alpha=0.3, axis='y', color='white')
            ax3.axhline(y=0, color='#2C3E50', linestyle='-', alpha=0.5, linewidth=1.5)
            ax3.set_ylim(-1.1, 1.1)
            ax3.tick_params(colors='#2C3E50')

            # Agregar valores en barras
            for bar, val in zip(bars, values):
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                offset = 0.05 if height >= 0 else -0.05
                ax3.text(bar.get_x() + bar.get_width()/2,
                        height + offset,
                        f'{val:.2f}',
                        ha='center', va=va,
                        fontweight='bold', fontsize=10, color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=colors[bars.index(bar)], alpha=0.8))

        # ==========================================
        # GRÁFICO 4: RECOMENDACIONES PROFESIONALES
        # ==========================================
        ax4 = plt.subplot(2, 2, 4)
        ax4.set_facecolor('#F8F9FA')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)

        ax4.text(0.5, 0.95, 'RECOMENDACIONES ESTRATÉGICAS',
                ha='center', va='top', fontsize=16, fontweight='bold', color='#2C3E50')

        # Analizar correlaciones para recomendaciones
        recommendations = []
        for var1, var2, corr in all_correlations[:10]:  # Top 10 correlaciones
            if abs(corr) > 0.5:
                var1_clean = var1.replace('_', ' ').title()
                var2_clean = var2.replace('_', ' ').title()
                strength = "fuerte" if abs(corr) > 0.7 else "moderada"
                direction = "positiva" if corr > 0 else "negativa"
                recommendations.append(f"• {var1_clean} ↔ {var2_clean}: {strength} {direction} ({corr:.2f})")

        # Agregar recomendaciones generales
        recommendations.extend([
            "",
            "ACCIONES RECOMENDADAS:",
            "• Variables con correlación positiva > 0.7: Ideales para predicción",
            "• Variables con correlación negativa: Considerar trade-offs",
            "• Variables independientes (< 0.3): Útiles para diversificar riesgos",
            "",
            "PRÓXIMOS PASOS:",
            "• Usar variables altamente correlacionadas para modelos predictivos",
            "• Investigar causas de correlaciones negativas inesperadas",
            "• Considerar variables independientes para estrategias de diversificación"
        ])

        y_position = 0.85
        for rec in recommendations:
            if rec.startswith("•") or rec.startswith("ACCIONES") or rec.startswith("PRÓXIMOS"):
                color = '#2C3E50' if rec.startswith("•") else '#E74C3C'
                fontweight = 'bold' if not rec.startswith("•") else 'normal'
                ax4.text(0.05, y_position, rec, fontsize=9, color=color, fontweight=fontweight)
            else:
                ax4.text(0.05, y_position, rec, fontsize=9, color='#34495E')
            y_position -= 0.06

        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.85, hspace=0.3, wspace=0.3)

        correlacion_file = OUTPUT_DIR / "correlacion_profesional.png"
        plt.savefig(correlacion_file, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
        logging.info(f"Análisis profesional de correlación guardado en {correlacion_file}")

        # Crear explicación profesional
        total_vars = len(correlation_matrix.columns)
        strong_correlations = sum(1 for _, _, corr in all_correlations if abs(corr) > 0.7)
        moderate_correlations = sum(1 for _, _, corr in all_correlations if 0.3 <= abs(corr) <= 0.7)

        # Crear explicación responsive basada en los resultados específicos
        explicacion_resultados = ""

        if top_5:
            explicacion_resultados += "🔍 TUS RESULTADOS ESPECÍFICOS:\n\n"
            for i, (var1, var2, corr) in enumerate(top_5[:3], 1):  # Mostrar top 3
                var1_clean = var1.replace('_', ' ').title()
                var2_clean = var2.replace('_', ' ').title()
                tipo_corr = "POSITIVA FUERTE" if corr > 0.7 else "NEGATIVA FUERTE" if corr < -0.7 else "MODERADA"
                explicacion_resultados += f"   {i}️⃣ {var1_clean} ↔ {var2_clean}: {corr:.3f} ({tipo_corr})\n"

            explicacion_resultados += "\n💡 INTERPRETACIÓN DE TUS DATOS:\n"
            for var1, var2, corr in top_5[:3]:
                var1_clean = var1.replace('_', ' ').title()
                var2_clean = var2.replace('_', ' ').title()
                if abs(corr) > 0.7:
                    if corr > 0:
                        explicacion_resultados += f"   • {var1_clean} y {var2_clean} van de la mano - cuando una mejora, la otra también\n"
                    else:
                        explicacion_resultados += f"   • {var1_clean} y {var2_clean} se compensan - cuando una sube, la otra baja\n"
                else:
                    explicacion_resultados += f"   • {var1_clean} y {var2_clean} tienen relación moderada - investigar más\n"
            explicacion_resultados += "\n"

        explanation = (
            "🔗 ANÁLISIS DE CORRELACIÓN AGRÍCOLA - RESULTADOS PERSONALIZADOS\n\n"
            "Imagínate que tienes un grupo de amigos y quieres saber cómo se relacionan entre sí. "
            "Eso es exactamente lo que hemos hecho con tus variables agrícolas.\n\n"
            "📊 LO QUE ANALIZAMOS EN TUS DATOS:\n"
            f"   • {total_vars} variables agrícolas de tu dataset\n"
            f"   • {strong_correlations} conexiones muy fuertes encontradas\n"
            f"   • {moderate_correlations} conexiones moderadas identificadas\n\n"
            f"{explicacion_resultados}"
            "🎯 GUÍA COMPLETA DE INTERPRETACIÓN:\n\n"
            "1️⃣ CORRELACIÓN POSITIVA FUERTE (+0.7 a +1.0)\n"
            "   • Significado: 'Cuando una sube, la otra también sube'\n"
            "   • Ejemplo: Superficie sembrada ↔ Producción\n"
            "   • Útil para: Predicciones confiables, planificación\n\n"
            "2️⃣ CORRELACIÓN NEGATIVA FUERTE (-1.0 a -0.7)\n"
            "   • Significado: 'Cuando una sube, la otra baja'\n"
            "   • Ejemplo: Lluvia excesiva ↔ Menos rendimiento\n"
            "   • Útil para: Encontrar equilibrios óptimos\n\n"
            "3️⃣ CORRELACIÓN MODERADA (±0.3 a ±0.7)\n"
            "   • Significado: 'Hay relación, pero no perfecta'\n"
            "   • Ejemplo: Fertilizante ↔ Producción (depende de otros factores)\n"
            "   • Útil para: Investigar causas adicionales\n\n"
            "4️⃣ CORRELACIÓN DÉBIL (±0.0 a ±0.3)\n"
            "   • Significado: 'Van por caminos separados'\n"
            "   • Ejemplo: Color del tractor ↔ Rendimiento del cultivo\n"
            "   • Útil para: Diversificar riesgos\n\n"
            "📈 TUS 4 GRÁFICOS EXPLICADOS:\n\n"
            "GRÁFICO 1 - GUÍA DE INTERPRETACIÓN:\n"
            "   • Diccionario visual de qué significa cada correlación\n"
            "   • Aprende a leer los números como un experto\n\n"
            "GRÁFICO 2 - MATRIZ DE CORRELACIÓN:\n"
            "   • Mapa completo de relaciones entre tus variables\n"
            "   • Rojo/azul = conexiones fuertes, gris = independientes\n\n"
            "GRÁFICO 3 - TOP RELACIONES:\n"
            "   • Las conexiones más importantes de tus datos\n"
            "   • Barras más altas = relaciones más relevantes\n\n"
            "GRÁFICO 4 - RECOMENDACIONES:\n"
            "   • Acciones específicas basadas en tus resultados\n\n"
            "💡 ¿QUÉ HACER CON TUS RESULTADOS?\n\n"
            "1️⃣ PARA TUS CORRELACIONES FUERTES:\n"
            "   • Úsalas como base para modelos predictivos\n"
            "   • Confía en estas relaciones para planificar\n\n"
            "2️⃣ PARA RELACIONES MODERADAS:\n"
            "   • Investiga qué otros factores influyen\n"
            "   • Busca condiciones donde la relación sea más fuerte\n\n"
            "3️⃣ PARA VARIABLES INDEPENDIENTES:\n"
            "   • Úsalas para crear estrategias diversificadas\n"
            "   • Reduce riesgos combinando variables no relacionadas\n\n"
            "⚠️ RECUERDA:\n"
            "   • Correlación NO significa causalidad\n"
            "   • Una tercera variable podría estar influyendo\n"
            "   • Las relaciones pueden cambiar con el tiempo\n\n"
            "🎯 PRÓXIMOS PASOS RECOMENDADOS:\n"
            "   • Usa las correlaciones > 0.7 para predicciones\n"
            "   • Investiga correlaciones inesperadas\n"
            "   • Actualiza este análisis periódicamente"
        )

        # Mostrar mensaje conciso primero
        resumen_rapido = (
            f"✅ ANÁLISIS DE CORRELACIÓN COMPLETADO\n\n"
            f"📊 Analizadas {total_vars} variables\n"
            f"🔗 {strong_correlations} conexiones fuertes encontradas\n"
            f"📈 {moderate_correlations} conexiones moderadas\n\n"
            f"💡 RESULTADOS PRINCIPALES:\n"
        )

        if top_5:
            for i, (var1, var2, corr) in enumerate(top_5[:3], 1):
                var1_clean = var1.replace('_', ' ').title()
                var2_clean = var2.replace('_', ' ').title()
                resumen_rapido += f"   {i}. {var1_clean} ↔ {var2_clean}: {corr:.3f}\n"

        resumen_rapido += f"\n📁 Gráfico guardado: {correlacion_file}\n\n¿Ver explicación completa?"

        # Preguntar si quiere ver explicación completa
        ver_completo = messagebox.askyesno("Análisis de Correlación",
                                          resumen_rapido)

        if ver_completo:
            # Crear ventana con explicación completa y scroll
            ventana_explicacion = tk.Toplevel(self.root)
            ventana_explicacion.title("📊 Explicación Completa - Análisis de Correlación")
            ventana_explicacion.geometry("900x700")

            # Frame principal
            frame_principal = tk.Frame(ventana_explicacion)
            frame_principal.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

            # Título
            titulo_label = tk.Label(frame_principal, text="🔗 ANÁLISIS DE CORRELACIÓN AGRÍCOLA DETALLADO",
                                   font=('Arial', 16, 'bold'), fg='#2563EB')
            titulo_label.pack(pady=(0, 20))

            # Text widget con scroll para la explicación completa
            frame_texto = tk.Frame(frame_principal)
            frame_texto.pack(fill=tk.BOTH, expand=True)

            scrollbar = tk.Scrollbar(frame_texto)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            texto_completo = tk.Text(frame_texto, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                                    font=('Arial', 11), padx=10, pady=10)
            texto_completo.pack(fill=tk.BOTH, expand=True)
            scrollbar.config(command=texto_completo.yview)

            # Insertar explicación completa
            texto_completo.insert(tk.END, explanation)
            texto_completo.config(state=tk.DISABLED)  # Hacer solo lectura

            # Botón cerrar
            boton_cerrar = tk.Button(frame_principal, text="Cerrar",
                                    command=ventana_explicacion.destroy,
                                    font=('Arial', 12, 'bold'),
                                    bg='#DC2626', fg='white', padx=20, pady=10)
            boton_cerrar.pack(pady=(20, 0))

            # Centrar ventana
            ventana_explicacion.transient(self.root)
            ventana_explicacion.grab_set()
        else:
            messagebox.showinfo("Análisis Completado",
                              f"✅ Análisis guardado exitosamente en:\n{correlacion_file}")

    def correlacion_sup_sembrada_cosechada(self):
        """
        Calcula y visualiza la correlación entre superficie sembrada y cosechada.

        Esta función permite seleccionar una provincia y analiza la relación
        entre lo sembrado y lo cosechado, proporcionando insights para optimizar
        la eficiencia agrícola.
        """
        if not self._check_csv_loaded():
            return
        if 'provincia' not in self.df.columns or 'sup_sembrada' not in self.df.columns or 'sup_cosechada' not in self.df.columns:
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

            # Crear gráfico de dispersión para mayor claridad
            plt.figure(figsize=(8, 6))
            plt.scatter(df_provincia['sup_sembrada'], df_provincia['sup_cosechada'], alpha=0.6, color='blue')
            plt.title(f'Correlación entre Superficie Sembrada y Cosechada\nProvincia: {selected_provincia}')
            plt.xlabel('Superficie Sembrada (hectáreas)')
            plt.ylabel('Superficie Cosechada (hectáreas)')
            plt.grid(True, alpha=0.3)

            # Agregar línea de tendencia
            z = np.polyfit(df_provincia['sup_sembrada'], df_provincia['sup_cosechada'], 1)
            p = np.poly1d(z)
            plt.plot(df_provincia['sup_sembrada'], p(df_provincia['sup_sembrada']), "r--", alpha=0.8)

            plt.suptitle("correlacion_sup_sembrada_cosechada", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
            output_file = OUTPUT_DIR / f"correlacion_{self.safe_file_name(selected_provincia)}.png"
            plt.savefig(output_file)
            plt.show()

            explanation = (
                f"La correlación entre la superficie sembrada y cosechada en la provincia {selected_provincia} es {correlacion:.2f}. "
                f"{suggestion}\n\n"
                f"📊 Datos analizados: {len(df_provincia)} registros\n"
                f"📈 Gráfico guardado en: {output_file}"
            )
            messagebox.showinfo("Correlación Sup. Sembrada-Sup. Cosechada", explanation)
        except Exception as e:
            logging.error(f"Error al calcular la correlación: {e}")
            messagebox.showerror("Error", f"Ocurrió un error al calcular la correlación: {e}")

    @staticmethod
    def get_correlation_suggestion(correlacion):
        """Devuelve una sugerencia basada en el valor de la correlación."""
        if correlacion >= 0.7:
            return ("Correlación alta positiva. Esto significa que cuando se siembra más superficie, "
                    "generalmente se cosecha más. Sugerencia: Mantener prácticas actuales y explorar "
                    "variedades de cultivos de alto rendimiento para maximizar la producción por hectárea.")
        elif correlacion <= 0.3:
            return ("Correlación baja. Esto indica que factores externos (clima, plagas, suelo) "
                    "pueden estar causando pérdidas entre siembra y cosecha. Sugerencia: Revisar "
                    "prácticas de cultivo, mejorar manejo de factores ambientales y considerar "
                    "técnicas de conservación.")
        else:
            return ("Correlación moderada. La relación entre siembra y cosecha es variable. "
                    "Sugerencia: Considerar diversificación de cultivos para reducir riesgos "
                    "y mejorar la estabilidad de la producción.")

    def produccion_total_por_provincia(self):
        """Genera una gráfica de la producción total por provincia."""
        if not self._check_csv_loaded():
            return
        if 'provincia' not in self.df.columns or 'produccion' not in self.df.columns or 'campana' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener las columnas 'provincia', 'produccion' y 'campana'.")
            return

        # Convertir la columna campana a string para evitar errores
        self.df['campana'] = self.df['campana'].astype(str)
        
        campanas = self.df['campana'].unique()
        if len(campanas) == 0:
            messagebox.showwarning("Advertencia", "No se encontraron campanas en el archivo CSV.")
            return

        campanas_limpias = [str(campana).strip() for campana in campanas if pd.notna(campana)]

        selected_campana = self.ask_option("Seleccionar Campaña", "Seleccione la campana:", campanas_limpias)
        if not selected_campana:
            return

        # Filtrar usando comparación directa en lugar de .str.strip()
        df_campana = self.df[self.df['campana'].astype(str).str.strip() == selected_campana]
        
        if df_campana.empty:
            messagebox.showwarning("Advertencia", "No se encontraron datos para la campana seleccionada.")
            return
            
        produccion_por_provincia = df_campana.groupby('provincia')['produccion'].sum().sort_values(ascending=False)

        if produccion_por_provincia.empty:
            messagebox.showwarning("Advertencia", "No se encontraron datos de producción para la campana seleccionada.")
            return

        title = f"Producción Total por Provincia - Campaña {selected_campana}"
        output_file = OUTPUT_DIR / f"produccion_por_provincia_{self.safe_file_name(selected_campana)}.png"
        Visualization.plot_bar_chart(produccion_por_provincia, title, "Provincias", "Producción [Tn]", output_file, "produccion_total_por_provincia")

        explanation = (
            "📊 PRODUCCIÓN POR PROVINCIA\n\n"
            "Esta gráfica muestra cuánto produce cada provincia en la campana seleccionada.\n\n"
            "🔍 ¿QUÉ VER?\n"
            "   • Provincias con barras más altas = más producción\n"
            "   • Provincias con barras más bajas = menos producción\n\n"
            "💡 ¿PARA QUÉ SIRVE?\n"
            "   • Saber dónde se produce más\n"
            "   • Decidir dónde invertir recursos\n"
            "   • Planificar distribución de ayuda agrícola"
        )
        messagebox.showinfo("Producción Total por Provincia", f"Gráfica guardada en {output_file}\n\n{explanation}")

    def evolucion_cultivos_por_campana(self):
        """Genera un gráfico de la evolución de los cultivos por campana con nombres limpios y legibles."""
        if not self._check_csv_loaded():
            return
        if 'campana' not in self.df.columns or 'cultivo' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener las columnas 'campana' y 'cultivo'.")
            return

        # Limpiar nombres de cultivos sin normalizar (mantener nombres originales legibles)
        df_trabajo = self.df.copy()
        df_trabajo['cultivo'] = df_trabajo['cultivo'].astype(str).str.strip().str.title()
        
        # Verificar columnas de interés
        columnas_interes = ['sup_sembrada', 'sup_cosechada', 'produccion']
        columnas_presentes = [col for col in columnas_interes if col in df_trabajo.columns]
        if not columnas_presentes:
            messagebox.showwarning("Advertencia", f"El archivo CSV debe contener al menos una de las columnas: {', '.join(columnas_interes)}.")
            return

        # Procesar fechas de campana de manera más robusta
        try:
            # Intentar diferentes formatos de fecha
            if df_trabajo['campana'].dtype == 'object':
                # Si es texto, intentar extraer el año
                df_trabajo['año'] = df_trabajo['campana'].astype(str).str.extract(r'(\d{4})').astype(float)
            else:
                # Si es numérico, usar directamente
                df_trabajo['año'] = pd.to_numeric(df_trabajo['campana'], errors='coerce')
            
            # Filtrar años válidos
            df_trabajo = df_trabajo.dropna(subset=['año'])
            df_trabajo['año'] = df_trabajo['año'].astype(int)
            
        except Exception as e:
            logging.error(f"Error procesando fechas de campana: {e}")
            messagebox.showerror("Error", "No se pudieron procesar las fechas de campana correctamente.")
            return

        if df_trabajo.empty:
            messagebox.showwarning("Advertencia", "No se encontraron datos válidos después del procesamiento.")
            return

        # Obtener cultivos únicos y limpios
        cultivos_disponibles = sorted(df_trabajo['cultivo'].dropna().unique())
        
        if len(cultivos_disponibles) == 0:
            messagebox.showwarning("Advertencia", "No se encontraron cultivos válidos en los datos.")
            return

        # Seleccionar cultivo
        cultivo_seleccionado = self.ask_option("Seleccionar Cultivo", "Seleccione el cultivo:", cultivos_disponibles)
        if not cultivo_seleccionado:
            return

        # Filtrar datos para el cultivo seleccionado
        df_filtrado = df_trabajo[df_trabajo['cultivo'] == cultivo_seleccionado]
        if df_filtrado.empty:
            messagebox.showwarning("Advertencia", f"No se encontraron datos para el cultivo seleccionado: {cultivo_seleccionado}.")
            return

        # Agrupar por año y sumar valores
        datos_agrupados = df_filtrado.groupby('año')[columnas_presentes].sum()

        if datos_agrupados.empty:
            messagebox.showwarning("Advertencia", "No hay datos suficientes para generar el gráfico.")
            return

        # Crear subgráficos si hay múltiples columnas
        if len(columnas_presentes) > 1:
            fig, axes = plt.subplots(len(columnas_presentes), 1, figsize=(14, 4*len(columnas_presentes)))
            if len(columnas_presentes) == 1:
                axes = [axes]

            for i, columna in enumerate(columnas_presentes):
                axes[i].plot(datos_agrupados.index, datos_agrupados[columna],
                           marker='o', linewidth=2, markersize=6, label=columna)
                axes[i].set_title(f'Evolución de {columna.replace("_", " ").title()} - {cultivo_seleccionado}')
                axes[i].set_xlabel('Año')
                axes[i].set_ylabel(columna.replace("_", " ").title())
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()

                # Agregar valores en los puntos
                for x, y in zip(datos_agrupados.index, datos_agrupados[columna]):
                    axes[i].annotate(f'{y:,.0f}', (x, y), textcoords="offset points",
                                   xytext=(0,10), ha='center', fontsize=8)
        else:
            # Un solo gráfico si hay una sola columna
            plt.figure(figsize=(14, 10))
            columna = columnas_presentes[0]
            plt.plot(datos_agrupados.index, datos_agrupados[columna],
                    marker='o', linewidth=3, markersize=8, color='steelblue')
            plt.title(f'Evolución de {columna.replace("_", " ").title()} - {cultivo_seleccionado}', fontsize=14)
            plt.xlabel('Año', fontsize=12)
            plt.ylabel(columna.replace("_", " ").title(), fontsize=12)
            plt.grid(True, alpha=0.3)

            # Agregar valores en los puntos
            for x, y in zip(datos_agrupados.index, datos_agrupados[columna]):
                plt.annotate(f'{y:,.0f}', (x, y), textcoords="offset points",
                           xytext=(0,10), ha='center', fontsize=10, fontweight='bold')

        plt.suptitle("evolucion_cultivos_por_campana", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
        plt.tight_layout()

        # Crear nombre de archivo seguro
        cultivo_filename = re.sub(r'[^\w\s-]', '', cultivo_seleccionado).strip().replace(' ', '_')
        evolucion_file = OUTPUT_DIR / f"evolucion_cultivo_{cultivo_filename}.png"
        plt.savefig(evolucion_file, dpi=300, bbox_inches='tight')
        plt.show()
        logging.info(f"Gráfica de evolución de cultivo guardada en {evolucion_file}")

        # Análisis adicional
        años_analizados = len(datos_agrupados)
        año_inicial = datos_agrupados.index.min()
        año_final = datos_agrupados.index.max()
        
        # Calcular tendencias
        tendencias = {}
        for columna in columnas_presentes:
            if len(datos_agrupados) > 1:
                valor_inicial = datos_agrupados[columna].iloc[0]
                valor_final = datos_agrupados[columna].iloc[-1]
                if valor_inicial > 0:
                    cambio_porcentual = ((valor_final - valor_inicial) / valor_inicial) * 100
                    tendencias[columna] = cambio_porcentual
                else:
                    tendencias[columna] = 0

        tendencias_texto = ""
        for columna, cambio in tendencias.items():
            direccion = "📈 Crecimiento" if cambio > 5 else "📉 Declive" if cambio < -5 else "➡️ Estable"
            tendencias_texto += f"   • {columna.replace('_', ' ').title()}: {direccion} ({cambio:+.1f}%)\n"

        explanation = (
            f"📈 EVOLUCIÓN DEL CULTIVO: {cultivo_seleccionado.upper()}\n\n"
            f"📅 Años estudiados: {año_inicial} - {año_final}\n"
            f"📊 Variables mostradas: {', '.join([col.replace('_', ' ').title() for col in columnas_presentes])}\n\n"
            f"📈 TENDENCIAS:\n{tendencias_texto}\n"
            f"💡 ¿QUÉ MUESTRA?\n"
            f"   • Cómo ha cambiado este cultivo a lo largo del tiempo\n"
            f"   • Si está creciendo, bajando o se mantiene igual\n"
            f"   • Los números exactos por cada año\n\n"
            f"📋 PARA QUÉ SIRVE:\n"
            f"   • Saber si vale la pena seguir sembrando este cultivo\n"
            f"   • Planificar siembras basadas en el pasado\n"
            f"   • Ver el impacto de clima o economía"
        )
        
        messagebox.showinfo("Evolución de Cultivo por Campaña", f"Gráfica guardada en {evolucion_file}\n\n{explanation}")

    def tendencias_produccion_por_cultivo(self):
        """Genera un gráfico de tendencias de producción por cultivo y campana mejorado."""
        if not self._check_csv_loaded():
            return
        if 'campana' not in self.df.columns or 'cultivo' not in self.df.columns or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener las columnas 'campana', 'cultivo' y 'produccion'.")
            return

        # Filtrar datos válidos
        df_valid = self.df.dropna(subset=['campana', 'cultivo', 'produccion']).copy()
        
        if len(df_valid) < 10:
            messagebox.showwarning("Advertencia", "No hay suficientes datos válidos para el análisis de tendencias.")
            return

        # Agrupar por cultivo y campana, sumando la producción
        df_grouped = df_valid.groupby(['cultivo', 'campana'])['produccion'].sum().reset_index()
        
        # Obtener los cultivos con mayor producción total para evitar amontonamiento
        produccion_total_por_cultivo = df_grouped.groupby('cultivo')['produccion'].sum().sort_values(ascending=False)
        
        # Seleccionar solo los top 8 cultivos para mejor visualización
        top_cultivos = produccion_total_por_cultivo.head(8).index.tolist()
        df_top = df_grouped[df_grouped['cultivo'].isin(top_cultivos)]
        
        # Crear subgráficos para mejor visualización
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Gráfico 1: Tendencias de los top 4 cultivos
        top_4_cultivos = top_cultivos[:4]
        for cultivo in top_4_cultivos:
            cultivo_data = df_top[df_top['cultivo'] == cultivo]
            ax1.plot(cultivo_data['campana'], cultivo_data['produccion'],
                    marker='o', linewidth=2, label=cultivo)
        
        ax1.set_title('Tendencias - Top 4 Cultivos por Producción')
        ax1.set_xlabel('Campaña')
        ax1.set_ylabel('Producción (toneladas)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Gráfico 2: Tendencias de los siguientes 4 cultivos
        if len(top_cultivos) > 4:
            next_4_cultivos = top_cultivos[4:8]
            for cultivo in next_4_cultivos:
                cultivo_data = df_top[df_top['cultivo'] == cultivo]
                ax2.plot(cultivo_data['campana'], cultivo_data['produccion'],
                        marker='s', linewidth=2, label=cultivo)
            
            ax2.set_title('Tendencias - Siguientes 4 Cultivos')
            ax2.set_xlabel('Campaña')
            ax2.set_ylabel('Producción (toneladas)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'Menos de 8 cultivos\ndisponibles',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Cultivos Adicionales')
        
        # Gráfico 3: Comparación de producción total por cultivo (barras)
        produccion_total_top = produccion_total_por_cultivo.head(10)
        bars = ax3.bar(range(len(produccion_total_top)), produccion_total_top.values,
                      color='lightblue', edgecolor='navy')
        ax3.set_title('Producción Total por Cultivo (Top 10)')
        ax3.set_xlabel('Cultivos')
        ax3.set_ylabel('Producción Total (toneladas)')
        ax3.set_xticks(range(len(produccion_total_top)))
        ax3.set_xticklabels(produccion_total_top.index, rotation=45, ha='right')
        
        # Agregar valores en las barras
        for bar, valor in zip(bars, produccion_total_top.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + valor*0.01,
                    f'{valor:,.0f}', ha='center', va='bottom', fontsize=8)
        
        # Gráfico 4: Evolución promedio de todos los cultivos
        evolucion_promedio = df_grouped.groupby('campana')['produccion'].mean()
        ax4.plot(evolucion_promedio.index, evolucion_promedio.values,
                marker='o', linewidth=3, color='red', label='Promedio General')
        ax4.fill_between(evolucion_promedio.index, evolucion_promedio.values, alpha=0.3, color='red')
        ax4.set_title('Evolución Promedio de Producción')
        ax4.set_xlabel('Campaña')
        ax4.set_ylabel('Producción Promedio (toneladas)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.suptitle("tendencias_produccion_por_cultivo", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
        plt.tight_layout()

        tendencias_file = OUTPUT_DIR / "tendencias_produccion.png"
        plt.savefig(tendencias_file, dpi=300, bbox_inches='tight')
        plt.show()
        logging.info(f"Gráfica de tendencias de producción guardada en {tendencias_file}")

        # Análisis adicional
        cultivo_mas_estable = None
        cultivo_mas_variable = None
        
        if len(df_top) > 0:
            # Calcular variabilidad (coeficiente de variación) para cada cultivo
            variabilidad_cultivos = {}
            for cultivo in top_cultivos:
                cultivo_data = df_top[df_top['cultivo'] == cultivo]['produccion']
                if len(cultivo_data) > 1:
                    cv = (cultivo_data.std() / cultivo_data.mean()) * 100
                    variabilidad_cultivos[cultivo] = cv
            
            if variabilidad_cultivos:
                cultivo_mas_estable = min(variabilidad_cultivos, key=variabilidad_cultivos.get)
                cultivo_mas_variable = max(variabilidad_cultivos, key=variabilidad_cultivos.get)

        explanation = (
            f"🌾 TENDENCIAS DE PRODUCCIÓN POR CULTIVO\n\n"
            f"📊 Datos revisados: {len(df_valid):,} registros\n"
            f"🌱 Tipos de cultivos: {len(df_valid['cultivo'].unique())}\n\n"
            f"🏆 CULTIVOS MÁS PRODUCTIVOS:\n"
            f"   1. {produccion_total_por_cultivo.index[0]}: {produccion_total_por_cultivo.iloc[0]:,.0f} ton\n"
            f"   2. {produccion_total_por_cultivo.index[1]}: {produccion_total_por_cultivo.iloc[1]:,.0f} ton\n"
            f"   3. {produccion_total_por_cultivo.index[2]}: {produccion_total_por_cultivo.iloc[2]:,.0f} ton\n\n"
            f"📈 ESTABILIDAD:\n"
            f"   🟢 Más estable: {cultivo_mas_estable if cultivo_mas_estable else 'No disponible'}\n"
            f"   🔴 Más variable: {cultivo_mas_variable if cultivo_mas_variable else 'No disponible'}\n\n"
            f"💡 ¿QUÉ MUESTRAN LOS GRÁFICOS?\n"
            f"   • Cómo cambia la producción de cada cultivo con el tiempo\n"
            f"   • Cuáles cultivos producen más\n"
            f"   • Cuáles son predecibles y cuáles cambian mucho\n\n"
            f"📋 USO PRÁCTICO:\n"
            f"   • Elegir cultivos confiables para sembrar\n"
            f"   • Diversificar para reducir riesgos\n"
            f"   • Planificar inversiones agrícolas"
        )
        
        messagebox.showinfo("Tendencias de Producción por Cultivo", f"Gráfica guardada en {tendencias_file}\n\n{explanation}")

    def modelos_predictivos(self):
        """Entrena y evalúa un modelo de regresión lineal con visualizaciones completas."""
        if not self._check_csv_loaded():
            return
        if 'sup_sembrada' not in self.df.columns or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener 'sup_sembrada' y 'produccion'.")
            return

        # Limpiar datos eliminando filas con NaN en las columnas relevantes
        df_clean = self.df.dropna(subset=['sup_sembrada', 'produccion']).copy()

        if len(df_clean) < 10:
            messagebox.showwarning("Advertencia", "Se necesitan al menos 10 registros válidos para el análisis predictivo.")
            return

        X = df_clean[['sup_sembrada']].values
        y = df_clean['produccion'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Crear visualizaciones completas
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Gráfico 1: Datos de entrenamiento y línea de regresión
        ax1.scatter(X_train, y_train, alpha=0.6, color='blue', label='Datos de entrenamiento', s=50)
        ax1.scatter(X_test, y_test, alpha=0.8, color='red', label='Datos de prueba', s=50)

        # Línea de regresión
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_plot = model.predict(X_plot)
        ax1.plot(X_plot, y_plot, 'g-', linewidth=3, label=f'Línea de regresión\ny = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')

        ax1.set_xlabel('Superficie Sembrada (hectáreas)')
        ax1.set_ylabel('Producción (toneladas)')
        ax1.set_title('Modelo de Regresión Lineal: Superficie vs Producción')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Gráfico 2: Predicciones vs Valores Reales
        ax2.scatter(y_test, y_pred, alpha=0.7, color='green', s=60, edgecolors='black')

        # Línea ideal (predicción perfecta)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Línea ideal (predicción perfecta)')

        # Línea de tendencia de las predicciones
        z = np.polyfit(y_test, y_pred, 1)
        p = np.poly1d(z)
        ax2.plot(y_test, p(y_test), "b--", alpha=0.8, linewidth=2, label='Tendencia de predicciones')

        ax2.set_xlabel('Producción Real (toneladas)')
        ax2.set_ylabel('Producción Predicha (toneladas)')
        ax2.set_title('Predicciones vs Valores Reales')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Gráfico 3: Distribución de errores (residuales)
        errores = y_test - y_pred
        ax3.hist(errores, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Sin error')
        ax3.axvline(errores.mean(), color='blue', linestyle='-', linewidth=2, label=f'Error promedio ({errores.mean():.1f})')

        ax3.set_xlabel('Error de Predicción (toneladas)')
        ax3.set_ylabel('Frecuencia')
        ax3.set_title('Distribución de Errores del Modelo')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Estadísticas de errores
        error_mean = np.mean(errores)
        error_std = np.std(errores)
        ax3.text(0.02, 0.98, f'Error promedio: {error_mean:.1f} ton\nDesviación: {error_std:.1f} ton',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Gráfico 4: Métricas de rendimiento del modelo
        metricas = ['R²', 'MSE', 'RMSE', 'MAE']
        valores = [r2, mse, rmse, np.mean(np.abs(errores))]

        colores_metricas = ['green' if r2 > 0.7 else 'orange' if r2 > 0.5 else 'red',
                           'red' if mse > np.var(y) else 'orange',
                           'red' if rmse > np.std(y) else 'orange',
                           'red' if np.mean(np.abs(errores)) > np.std(y) else 'orange']

        bars = ax4.bar(metricas, valores, color=colores_metricas, alpha=0.8, edgecolor='black')

        # Agregar valores en las barras
        for bar, val in zip(bars, valores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax4.set_ylabel('Valor de la Métrica')
        ax4.set_title('Métricas de Rendimiento del Modelo')
        ax4.grid(True, alpha=0.3, axis='y')

        # Agregar líneas de referencia
        ax4.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='R² bueno (>0.8)')
        ax4.axhline(y=np.var(y), color='red', linestyle='--', alpha=0.7, label='MSE referencia (varianza total)')
        ax4.legend(fontsize=8)

        plt.suptitle("modelos_predictivos", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
        plt.tight_layout()

        # Guardar gráfico
        output_file = OUTPUT_DIR / "modelo_predictivo_completo.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()

        # Análisis adicional del modelo
        pendiente = model.coef_[0]
        intercepto = model.intercept_

        # Calcular predicciones para valores extremos
        sup_min = df_clean['sup_sembrada'].min()
        sup_max = df_clean['sup_sembrada'].max()
        prod_pred_min = model.predict([[sup_min]])[0]
        prod_pred_max = model.predict([[sup_max]])[0]

        explanation = (
            f"📈 MODELO PREDICTIVO DE REGRESIÓN LINEAL COMPLETO\n\n"
            f"Este análisis crea un modelo matemático que predice la producción agrícola "
            f"basándose únicamente en la superficie sembrada.\n\n"
            f"🔢 ECUACIÓN DEL MODELO:\n"
            f"   Producción = {pendiente:.2f} × Superficie_Sembrada + {intercepto:.2f}\n\n"
            f"📊 MÉTRICAS DE RENDIMIENTO:\n"
            f"   • R² (precisión): {r2:.3f} - {'Excelente (>0.8)' if r2 > 0.8 else 'Buena (0.6-0.8)' if r2 > 0.6 else 'Aceptable (0.3-0.6)' if r2 > 0.3 else 'Pobre (<0.3)'}\n"
            f"   • MSE (error cuadrático medio): {mse:.0f} toneladas²\n"
            f"   • RMSE (error típico): {rmse:.0f} toneladas\n"
            f"   • MAE (error absoluto medio): {np.mean(np.abs(errores)):.0f} toneladas\n\n"
            f"📈 PREDICCIONES EXTREMAS:\n"
            f"   • Con {sup_min:,.0f} ha sembradas → {prod_pred_min:,.0f} ton predichas\n"
            f"   • Con {sup_max:,.0f} ha sembradas → {prod_pred_max:,.0f} ton predichas\n\n"
            f"💡 INTERPRETACIÓN DE GRÁFICOS:\n"
            f"   • Gráfico 1: Muestra cómo se ajusta la línea de regresión a los datos\n"
            f"   • Gráfico 2: Compara predicciones vs realidad (puntos cerca de línea roja = buenas predicciones)\n"
            f"   • Gráfico 3: Distribución de errores (curva centrada en cero = buen modelo)\n"
            f"   • Gráfico 4: Métricas cuantitativas del rendimiento\n\n"
            f"⚠️ LIMITACIONES:\n"
            f"   • Solo usa una variable predictora (superficie sembrada)\n"
            f"   • No considera factores como clima, suelo, variedad de cultivo\n"
            f"   • Es un modelo lineal simple (relaciones complejas pueden no capturarse)\n\n"
            f"📋 RECOMENDACIONES:\n"
            f"   • {'El modelo es confiable' if r2 > 0.7 else 'Considerar más variables o modelos complejos' if r2 > 0.5 else 'Revisar datos o cambiar enfoque'}\n"
            f"   • Usar para estimaciones rápidas de producción\n"
            f"   • Combinar con otros factores para predicciones más precisas\n\n"
            f"📊 Gráfico guardado en: {output_file}"
        )

        messagebox.showinfo("Modelo Predictivo Completo", explanation)

    def clasificacion_cultivos(self):
        """Analiza y clasifica cultivos según características de producción."""
        columnas_requeridas = ['cultivo']
        columnas_opcionales = ['sup_sembrada', 'sup_cosechada', 'produccion', 'rendimiento', 'provincia']
        
        if not self._check_csv_loaded():
            return
        if 'cultivo' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener la columna 'cultivo'.")
            return

        # Filtrar datos válidos
        df_valid = self.df.dropna(subset=['cultivo']).copy()
        
        if len(df_valid) < 10:
            messagebox.showwarning("Advertencia", "No hay suficientes datos válidos para realizar la clasificación.")
            return

        # Análisis descriptivo de cultivos
        total_cultivos = len(df_valid['cultivo'].unique())
        cultivos_mas_comunes = df_valid['cultivo'].value_counts().head(10)
        
        # Crear visualización
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Gráfico 1: Distribución de cultivos (top 10)
        cultivos_mas_comunes.plot(kind='bar', ax=ax1, color='lightgreen')
        ax1.set_title('Top 10 Cultivos Más Frecuentes')
        ax1.set_xlabel('Tipo de Cultivo')
        ax1.set_ylabel('Cantidad de Registros')
        ax1.tick_params(axis='x', rotation=45)
        
        # Agregar valores en las barras
        for i, v in enumerate(cultivos_mas_comunes.values):
            ax1.text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')

        # Gráfico 2: Producción promedio por cultivo (si está disponible)
        if 'produccion' in df_valid.columns:
            produccion_por_cultivo = df_valid.groupby('cultivo')['produccion'].mean().sort_values(ascending=False).head(10)
            produccion_por_cultivo.plot(kind='bar', ax=ax2, color='orange')
            ax2.set_title('Producción Promedio por Cultivo (Top 10)')
            ax2.set_xlabel('Tipo de Cultivo')
            ax2.set_ylabel('Producción Promedio (toneladas)')
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'Datos de producción\nno disponibles',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Producción por Cultivo')

        # Gráfico 3: Superficie sembrada promedio por cultivo (si está disponible)
        if 'sup_sembrada' in df_valid.columns:
            superficie_por_cultivo = df_valid.groupby('cultivo')['sup_sembrada'].mean().sort_values(ascending=False).head(10)
            superficie_por_cultivo.plot(kind='bar', ax=ax3, color='skyblue')
            ax3.set_title('Superficie Sembrada Promedio por Cultivo (Top 10)')
            ax3.set_xlabel('Tipo de Cultivo')
            ax3.set_ylabel('Superficie Promedio (hectáreas)')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'Datos de superficie\nno disponibles',
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Superficie Sembrada por Cultivo')

        # Gráfico 4: Distribución por provincia (si está disponible)
        if 'provincia' in df_valid.columns:
            cultivos_por_provincia = df_valid.groupby('provincia')['cultivo'].nunique().sort_values(ascending=False).head(10)
            cultivos_por_provincia.plot(kind='bar', ax=ax4, color='lightcoral')
            ax4.set_title('Diversidad de Cultivos por Provincia (Top 10)')
            ax4.set_xlabel('Provincia')
            ax4.set_ylabel('Cantidad de Tipos de Cultivos')
            ax4.tick_params(axis='x', rotation=45)
        else:
            # Gráfico de torta de cultivos principales
            cultivos_principales = df_valid['cultivo'].value_counts().head(8)
            otros = df_valid['cultivo'].value_counts().iloc[8:].sum()
            if otros > 0:
                cultivos_principales['Otros'] = otros
            
            ax4.pie(cultivos_principales.values, labels=cultivos_principales.index, autopct='%1.1f%%')
            ax4.set_title('Distribución de Cultivos Principales')

        plt.suptitle("clasificacion_cultivos", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
        plt.tight_layout()
        
        # Guardar gráfico
        clasificacion_file = OUTPUT_DIR / "clasificacion_cultivos.png"
        plt.savefig(clasificacion_file, dpi=300, bbox_inches='tight')
        plt.show()
        logging.info(f"Análisis de clasificación de cultivos guardado en {clasificacion_file}")

        # Estadísticas adicionales
        estadisticas_adicionales = ""
        if 'produccion' in df_valid.columns:
            cultivo_mas_productivo = df_valid.groupby('cultivo')['produccion'].mean().idxmax()
            produccion_maxima = df_valid.groupby('cultivo')['produccion'].mean().max()
            estadisticas_adicionales += f"\n🏆 Cultivo más productivo: {cultivo_mas_productivo} ({produccion_maxima:.0f} ton promedio)"
        
        if 'sup_sembrada' in df_valid.columns:
            cultivo_mayor_superficie = df_valid.groupby('cultivo')['sup_sembrada'].mean().idxmax()
            superficie_maxima = df_valid.groupby('cultivo')['sup_sembrada'].mean().max()
            estadisticas_adicionales += f"\n🌾 Cultivo con mayor superficie: {cultivo_mayor_superficie} ({superficie_maxima:.0f} ha promedio)"

        explanation = (
            f"📊 CLASIFICACIÓN Y ANÁLISIS DE CULTIVOS\n\n"
            f"🔍 Datos analizados: {len(df_valid):,} registros de cultivos\n"
            f"🌱 Total de tipos de cultivos: {total_cultivos}\n\n"
            f"📈 Top 3 Cultivos Más Frecuentes:\n"
            f"   1. {cultivos_mas_comunes.index[0]}: {cultivos_mas_comunes.iloc[0]} registros\n"
            f"   2. {cultivos_mas_comunes.index[1]}: {cultivos_mas_comunes.iloc[1]} registros\n"
            f"   3. {cultivos_mas_comunes.index[2]}: {cultivos_mas_comunes.iloc[2]} registros\n"
            f"{estadisticas_adicionales}\n\n"
            f"💡 ¿Qué muestra este análisis?\n"
            f"   • Identifica qué cultivos son más comunes en tu dataset\n"
            f"   • Compara la productividad promedio entre diferentes cultivos\n"
            f"   • Analiza qué cultivos requieren más superficie para sembrar\n"
            f"   • Muestra la diversidad de cultivos por región\n\n"
            f"📋 Utilidad práctica:\n"
            f"   • Planificación de siembra basada en cultivos exitosos\n"
            f"   • Identificación de oportunidades de diversificación\n"
            f"   • Comparación de eficiencia entre cultivos\n"
            f"   • Análisis de especialización regional"
        )
        
        messagebox.showinfo("Clasificación de Cultivos", explanation)

    def analisis_riesgos(self):
        """Realiza un análisis de riesgos agrícolas identificando zonas de alta, media y baja producción por provincia, cultivo o global."""
        columnas_requeridas = ['produccion']
        columnas_opcionales = ['provincia', 'campana', 'departamento', 'cultivo']

        # Verificar columnas requeridas
        if not self._check_csv_loaded():
            return
        if not all(col in self.df.columns for col in columnas_requeridas):
            messagebox.showwarning("Advertencia", "El DataFrame debe contener la columna 'produccion'.")
            return

        # Preguntar tipo de análisis
        tipos_analisis = ["Análisis Global", "Por Provincia", "Por Cultivo"]
        tipo_seleccionado = self.ask_option("Tipo de Análisis de Riesgos",
                                          "Seleccione el tipo de análisis de riesgos:", tipos_analisis)
        if not tipo_seleccionado:
            return

        # Filtrar datos según selección
        df_valid = self.df[self.df['produccion'].notna() & (self.df['produccion'] > 0)].copy()

        if tipo_seleccionado == "Por Provincia":
            if 'provincia' not in df_valid.columns:
                messagebox.showwarning("Advertencia", "La columna 'provincia' no está disponible para análisis por provincia.")
                return
            provincias = sorted(df_valid['provincia'].dropna().unique())
            provincia_seleccionada = self.ask_option("Seleccionar Provincia",
                                                   "Seleccione la provincia para analizar:", provincias)
            if not provincia_seleccionada:
                return
            df_valid = df_valid[df_valid['provincia'] == provincia_seleccionada]
            titulo_base = f"Análisis de Riesgos - Provincia: {provincia_seleccionada}"

        elif tipo_seleccionado == "Por Cultivo":
            if 'cultivo' not in df_valid.columns:
                messagebox.showwarning("Advertencia", "La columna 'cultivo' no está disponible para análisis por cultivo.")
                return
            cultivos = sorted(df_valid['cultivo'].dropna().unique())
            cultivo_seleccionado = self.ask_option("Seleccionar Cultivo",
                                                 "Seleccione el cultivo para analizar:", cultivos)
            if not cultivo_seleccionado:
                return
            df_valid = df_valid[df_valid['cultivo'] == cultivo_seleccionado]
            titulo_base = f"Análisis de Riesgos - Cultivo: {cultivo_seleccionado}"

        else:  # Análisis Global
            titulo_base = "Análisis de Riesgos - Global"

        if len(df_valid) < 10:
            messagebox.showwarning("Advertencia", f"No hay suficientes datos válidos para realizar el análisis de riesgos ({len(df_valid)} registros, mínimo 10 requeridos).")
            return

        # Limitar a una muestra para evitar consumo excesivo de RAM
        if len(df_valid) > 5000:
            df_valid = df_valid.sample(n=5000, random_state=42)
            logging.info("Muestra limitada a 5000 filas para análisis de riesgos.")

        # Obtener información temporal si está disponible
        campanas_info = ""
        if 'campana' in df_valid.columns:
            campanas_unicas = sorted(df_valid['campana'].dropna().unique())
            if len(campanas_unicas) > 0:
                primera_campana = campanas_unicas[0]
                ultima_campana = campanas_unicas[-1]
                total_campanas = len(campanas_unicas)
                campanas_info = f"📅 Período analizado: {primera_campana} - {ultima_campana} ({total_campanas} campanas)\n"

        # Calcular estadísticas básicas de producción
        produccion_values = df_valid['produccion'].values
        media_produccion = np.mean(produccion_values)
        std_produccion = np.std(produccion_values)
        min_produccion = np.min(produccion_values)
        max_produccion = np.max(produccion_values)

        # Definir umbrales de riesgo basados en percentiles
        percentil_33 = np.percentile(produccion_values, 33)
        percentil_66 = np.percentile(produccion_values, 66)

        # Clasificar riesgos
        def clasificar_riesgo(produccion):
            if produccion <= percentil_33:
                return 'Alto Riesgo'
            elif produccion <= percentil_66:
                return 'Riesgo Medio'
            else:
                return 'Bajo Riesgo'

        # Aplicar clasificación
        df_valid['Nivel_Riesgo'] = df_valid['produccion'].apply(clasificar_riesgo)
        
        # Contar casos por nivel de riesgo
        conteo_riesgos = df_valid['Nivel_Riesgo'].value_counts()

        # Análisis por provincia
        zonas_alto_riesgo = []
        zonas_medio_riesgo = []
        zonas_bajo_riesgo = []

        # Definir produccion_por_provincia si 'provincia' está disponible
        if 'provincia' in df_valid.columns:
            produccion_por_provincia = df_valid.groupby('provincia').agg({
                'produccion': ['mean', 'count'],
                'Nivel_Riesgo': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Sin datos'
            }).round(2)

            produccion_por_provincia.columns = ['Produccion_Promedio', 'Cantidad_Registros', 'Nivel_Riesgo_Predominante']
            produccion_por_provincia = produccion_por_provincia.reset_index()

            # Clasificar provincias por nivel de riesgo predominante
            for _, row in produccion_por_provincia.iterrows():
                provincia = row['provincia']
                nivel = row['Nivel_Riesgo_Predominante']
                prod_prom = row['Produccion_Promedio']

                if nivel == 'Alto Riesgo':
                    zonas_alto_riesgo.append(f"{provincia} ({prod_prom:.0f} ton promedio)")
                elif nivel == 'Riesgo Medio':
                    zonas_medio_riesgo.append(f"{provincia} ({prod_prom:.0f} ton promedio)")
                else:
                    zonas_bajo_riesgo.append(f"{provincia} ({prod_prom:.0f} ton promedio)")

        # Crear visualización mejorada
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Gráfico 1: Histograma de producción con umbrales de riesgo
        ax1.hist(produccion_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(percentil_33, color='red', linestyle='--', linewidth=2, label=f'Alto Riesgo (≤{percentil_33:.0f})')
        ax1.axvline(percentil_66, color='orange', linestyle='--', linewidth=2, label=f'Riesgo Medio (≤{percentil_66:.0f})')
        ax1.axvline(media_produccion, color='green', linestyle='-', linewidth=2, label=f'Media ({media_produccion:.0f})')
        ax1.set_xlabel('Producción (toneladas)')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución de Producción con Umbrales de Riesgo')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Gráfico 2: Gráfico de barras por nivel de riesgo
        colores = ['red', 'orange', 'green']
        bars = ax2.bar(conteo_riesgos.index, conteo_riesgos.values, color=colores)
        ax2.set_xlabel('Nivel de Riesgo')
        ax2.set_ylabel('Cantidad de Casos')
        ax2.set_title('Distribución por Nivel de Riesgo')
        
        # Agregar valores en las barras
        for bar, valor in zip(bars, conteo_riesgos.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(valor), ha='center', va='bottom', fontweight='bold')

        # Gráfico 3: Producción por provincia si está disponible (si hay muchas, se muestran las Top 15)
        if 'provincia' in df_valid.columns:
            top_n = 15
            produccion_por_provincia_sorted = produccion_por_provincia.sort_values('Produccion_Promedio')
            if len(produccion_por_provincia_sorted) > top_n:
                produccion_por_provincia_sorted = produccion_por_provincia_sorted.tail(top_n)

            colores_provincias = ['red' if x == 'Alto Riesgo' else 'orange' if x == 'Riesgo Medio' else 'green'
                                  for x in produccion_por_provincia_sorted['Nivel_Riesgo_Predominante']]
            
            bars = ax3.barh(produccion_por_provincia_sorted['provincia'],
                            produccion_por_provincia_sorted['Produccion_Promedio'],
                            color=colores_provincias)
            ax3.set_xlabel('Producción Promedio (toneladas)')
            ax3.set_ylabel('Provincia')
            ax3.set_title('Producción Promedio por Provincia (Top 15)')
            ax3.grid(True, alpha=0.3)
        else:
            # Gráfico de dispersión alternativo
            colores_scatter = {'Alto Riesgo': 'red', 'Riesgo Medio': 'orange', 'Bajo Riesgo': 'green'}
            for nivel in df_valid['Nivel_Riesgo'].unique():
                subset = df_valid[df_valid['Nivel_Riesgo'] == nivel]
                ax3.scatter(range(len(subset)), subset['produccion'],
                            c=colores_scatter[nivel], label=nivel, alpha=0.6)
            ax3.set_xlabel('Índice de Registro')
            ax3.set_ylabel('Producción (toneladas)')
            ax3.set_title('Producción por Registro Clasificada por Riesgo')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Gráfico 4: Gráfico de torta
        ax4.pie(conteo_riesgos.values, labels=conteo_riesgos.index, colors=colores,
                autopct='%1.1f%%', startangle=90)
        ax4.set_title('Proporción de Niveles de Riesgo')

        plt.suptitle(f"analisis_riesgos - {tipo_seleccionado.replace(' ', '_').lower()}", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
        plt.tight_layout()

        # Crear nombre de archivo basado en el tipo de análisis
        if tipo_seleccionado == "Análisis Global":
            filename_suffix = "global"
        elif tipo_seleccionado == "Por Provincia":
            filename_suffix = f"provincia_{provincia_seleccionada.replace(' ', '_')}"
        else:  # Por Cultivo
            filename_suffix = f"cultivo_{cultivo_seleccionado.replace(' ', '_')}"

        riesgo_file = OUTPUT_DIR / f"analisis_riesgos_{filename_suffix}.png"
        plt.savefig(riesgo_file, dpi=300, bbox_inches='tight')
        plt.show()
        logging.info(f"Análisis de riesgos guardado en {riesgo_file}")

        explanation = ""

        # Generar mapa de riesgos por provincia si es análisis global y hay coordenadas
        if tipo_seleccionado == "Análisis Global" and 'provincia' in df_valid.columns and 'Latitude' in df_valid.columns and 'Longitude' in df_valid.columns:
            # Calcular coordenadas promedio por provincia
            coords_por_provincia = df_valid.groupby('provincia').agg({
                'Latitude': 'mean',
                'Longitude': 'mean'
            }).reset_index()

            # Unir con produccion_por_provincia
            mapa_data = produccion_por_provincia.merge(coords_por_provincia, on='provincia', how='left')

            # Crear mapa
            if not mapa_data.empty and mapa_data['Latitude'].notna().any():
                centro = [mapa_data['Latitude'].mean(), mapa_data['Longitude'].mean()]
                mapa_riesgos = folium.Map(location=centro, zoom_start=6)

                color_map = {
                    'Alto Riesgo': 'red',
                    'Riesgo Medio': 'orange',
                    'Bajo Riesgo': 'green'
                }

                for _, row in mapa_data.iterrows():
                    if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                        folium.CircleMarker(
                            location=[row['Latitude'], row['Longitude']],
                            radius=10,
                            popup=f"Provincia: {row['provincia']}<br>Producción Promedio: {row['Produccion_Promedio']:.0f} ton<br>Nivel de Riesgo: {row['Nivel_Riesgo_Predominante']}",
                            color=color_map.get(row['Nivel_Riesgo_Predominante'], 'gray'),
                            fill=True,
                            fill_color=color_map.get(row['Nivel_Riesgo_Predominante'], 'gray'),
                            fill_opacity=0.7
                        ).add_to(mapa_riesgos)

                mapa_riesgos_file = OUTPUT_DIR / "mapa_riesgos_provincias.html"
                mapa_riesgos.save(mapa_riesgos_file)
                logging.info(f"Mapa de riesgos por provincia guardado en {mapa_riesgos_file}")

                # Abrir el mapa
                webbrowser.open(mapa_riesgos_file.resolve().as_uri())

                # Actualizar explicación
                explanation += f"\n\n🗺️ MAPA DE RIESGOS POR PROVINCIA:\n   • Verde: Bajo Riesgo\n   • Naranja: Riesgo Medio\n   • Rojo: Alto Riesgo\n   • Archivo: {mapa_riesgos_file}"
            else:
                explanation += "\n\n⚠️ No se pudo generar el mapa de riesgos por provincia (faltan coordenadas)."

        # Asignar clasificación al DataFrame principal (solo para análisis global)
        if tipo_seleccionado == "Análisis Global":
            self.df.loc[df_valid.index, 'Nivel_Riesgo'] = df_valid['Nivel_Riesgo']

        # Crear reporte detallado
        porcentaje_alto = (conteo_riesgos.get('Alto Riesgo', 0) / len(df_valid)) * 100
        porcentaje_medio = (conteo_riesgos.get('Riesgo Medio', 0) / len(df_valid)) * 100
        porcentaje_bajo = (conteo_riesgos.get('Bajo Riesgo', 0) / len(df_valid)) * 100

        # No mostrar zonas_info separadamente, solo en la interpretación

        explanation = (
            f"📊 {titulo_base.upper()}\n\n"
            f"{campanas_info}"
            f"🔍 Datos analizados: {len(df_valid):,} registros de producción\n\n"
            f"📈 Estadísticas de Producción:\n"
            f"   • Producción mínima: {min_produccion:,.0f} toneladas\n"
            f"   • Producción máxima: {max_produccion:,.0f} toneladas\n"
            f"   • Producción promedio: {media_produccion:,.0f} toneladas\n\n"
            f"⚠️ Clasificación de Riesgos:\n"
            f"   🔴 ALTO RIESGO (≤{percentil_33:.0f} ton): {conteo_riesgos.get('Alto Riesgo', 0)} casos ({porcentaje_alto:.1f}%)\n"
            f"   🟡 RIESGO MEDIO ({percentil_33:.0f}-{percentil_66:.0f} ton): {conteo_riesgos.get('Riesgo Medio', 0)} casos ({porcentaje_medio:.1f}%)\n"
            f"   🟢 BAJO RIESGO (>{percentil_66:.0f} ton): {conteo_riesgos.get('Bajo Riesgo', 0)} casos ({porcentaje_bajo:.1f}%)\n\n"
            f"💡 Interpretación:\n"
            f"   • Las provincias de ALTO RIESGO ({', '.join(zonas_alto_riesgo) if zonas_alto_riesgo else 'ninguna'}) requieren atención inmediata\n"
            f"   • Las provincias de RIESGO MEDIO ({', '.join(zonas_medio_riesgo) if zonas_medio_riesgo else 'ninguna'}) necesitan monitoreo\n"
            f"   • Las provincias de BAJO RIESGO ({', '.join(zonas_bajo_riesgo) if zonas_bajo_riesgo else 'ninguna'}) son las más productivas\n\n"
            f"📋 Recomendaciones:\n"
            f"   • Investigar causas en zonas de alto riesgo (clima, suelo, plagas)\n"
            f"   • Implementar mejores prácticas en zonas de riesgo medio\n"
            f"   • Replicar estrategias exitosas de zonas de bajo riesgo"
        ) + explanation

        messagebox.showinfo(titulo_base, explanation)


    def prediccion_tendencias_ia(self):
        """Realiza predicción avanzada de tendencias agrícolas usando múltiples algoritmos de IA con optimización de hiperparámetros."""
        if not self._check_csv_loaded():
            return
        if 'campana' not in self.df.columns or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener las columnas 'campana' y 'produccion'.")
            return

        # Preparar datos
        df_trabajo = self.df.dropna(subset=['campana', 'produccion']).copy()
        if len(df_trabajo) < 10:
            messagebox.showwarning("Advertencia", "Se necesitan al menos 10 registros para el análisis predictivo.")
            return

        # Convertir campana a valores numéricos para el análisis (manejar formato "2023/2024")
        try:
            # Intentar convertir campanas al formato usado en otras funciones
            df_trabajo['año_numerico'] = df_trabajo['campana'].astype(str).str.split('/').str[0].astype(int)
        except (ValueError, AttributeError):
            # Si no funciona, intentar conversión directa
            df_trabajo['año_numerico'] = pd.to_numeric(df_trabajo['campana'], errors='coerce')

        # Filtrar valores válidos
        df_trabajo = df_trabajo.dropna(subset=['año_numerico'])
        if len(df_trabajo) < 10:
            messagebox.showwarning("Advertencia", "No hay suficientes datos válidos después del procesamiento de las campanas.")
            return
        df_trabajo['año_numerico'] = df_trabajo['año_numerico'].astype(int)

        # Limitar el tamaño del dataset para evitar tiempos de procesamiento excesivos
        max_samples = 1000
        if len(df_trabajo) > max_samples:
            df_trabajo = df_trabajo.sample(n=max_samples, random_state=42)
            logging.info(f"Dataset limitado a {max_samples} muestras para optimización de rendimiento.")

        X = df_trabajo[['año_numerico']].values
        y = df_trabajo['produccion'].values

        # Escalar características para mejor rendimiento
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42)

        # Usar Regresión Lineal para predicción de tendencias (asegura variación en predicciones)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluar en conjunto de prueba
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Calcular métricas
        mse = mean_squared_error(y_test_orig, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_orig - y_pred))
        r2 = r2_score(y_test_orig, y_pred)

        best_model = model
        best_model_name = 'Regresión Lineal'
        best_score = r2

        print(f"✅ Regresión Lineal: R² = {r2:.3f}, RMSE = {rmse:.2f}")

        # Crear visualización comparativa
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Gráfico 1: Rendimiento del modelo
        width = 0.35
        bars1 = ax1.bar(0, r2, width, label='R²', color='skyblue', alpha=0.8)
        ax1.set_ylabel('Coeficiente de Determinación (R²)', color='skyblue')
        ax1.set_title('Rendimiento del Modelo de Regresión Lineal')
        ax1.set_xticks([])
        ax1.legend(loc='upper left')

        ax1_twin = ax1.twinx()
        bars2 = ax1_twin.bar(width, rmse, width, label='RMSE', color='orange', alpha=0.8)
        ax1_twin.set_ylabel('Error Cuadrático Medio (RMSE)', color='orange')
        ax1_twin.legend(loc='upper right')

        # Agregar valores en barras
        ax1.text(bars1.get_x() + bars1.get_width()/2, bars1.get_height() + 0.01,
                f'{r2:.3f}', ha='center', va='bottom', fontsize=8)

        ax1_twin.text(bars2.get_x() + bars2.get_width()/2, bars2.get_height() + rmse*0.02,
                     f'{rmse:.0f}', ha='center', va='bottom', fontsize=8)

        # Gráfico 2: Predicciones vs Valores Reales
        ax2.scatter(y_test_orig, y_pred, alpha=0.6, color='green', s=50)
        ax2.plot([y_test_orig.min(), y_test_orig.max()],
                [y_test_orig.min(), y_test_orig.max()],
                'r--', linewidth=2, label='Línea ideal')
        ax2.set_xlabel('Producción Real (toneladas)')
        ax2.set_ylabel('Producción Predicha (toneladas)')
        ax2.set_title(f'Predicciones vs Realidad - {best_model_name}')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Agregar línea de tendencia
        z = np.polyfit(y_test_orig, y_pred, 1)
        p = np.poly1d(z)
        ax2.plot(y_test_orig, p(y_test_orig), "b--", alpha=0.8, label='Tendencia')

        # Gráfico 3: Serie temporal con predicciones
        años_ordenados = np.sort(df_trabajo['año_numerico'].unique())
        produccion_real = df_trabajo.groupby('año_numerico')['produccion'].mean()

        ax3.plot(produccion_real.index, produccion_real.values,
                'o-', linewidth=2, label='Producción Real', color='blue')

        # Generar predicciones para años futuros
        años_futuros = np.arange(años_ordenados.max() + 1, años_ordenados.max() + 6)
        X_futuro = scaler_X.transform(años_futuros.reshape(-1, 1))
        y_futuro_scaled = best_model.predict(X_futuro)
        y_futuro = scaler_y.inverse_transform(y_futuro_scaled.reshape(-1, 1)).ravel()

        ax3.plot(años_futuros, y_futuro, 'r--o', linewidth=2,
                label='Predicción IA (5 años)', markersize=6)

        ax3.set_xlabel('Campaña')
        ax3.set_ylabel('Producción Promedio (toneladas)')
        ax3.set_title('Tendencias Históricas y Predicciones Futuras')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Gráfico 4: Distribución de errores
        errores = y_test_orig - y_pred
        ax4.hist(errores, bins=20, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Sin error')
        ax4.set_xlabel('Error de Predicción (toneladas)')
        ax4.set_ylabel('Frecuencia')
        ax4.set_title('Distribución de Errores de Predicción')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Estadísticas de errores
        error_mean = np.mean(errores)
        error_std = np.std(errores)
        ax4.text(0.02, 0.98, f'Error promedio: {error_mean:.1f} ton\nDesviación: {error_std:.1f} ton',
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.suptitle("prediccion_tendencias_ia", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
        plt.tight_layout()

        # Guardar gráfico
        output_file = OUTPUT_DIR / "prediccion_tendencias_ia_avanzada.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        logging.info(f"Análisis predictivo avanzado guardado en {output_file}")

        # Crear reporte detallado

        # Calcular estadísticas adicionales
        total_datos = len(df_trabajo)
        años_unicos = len(df_trabajo['año_numerico'].unique())
        produccion_total = df_trabajo['produccion'].sum()

        # Análisis de tendencias
        años_sorted = sorted(df_trabajo['año_numerico'].unique())
        if len(años_sorted) > 1:
            prod_inicial = df_trabajo[df_trabajo['año_numerico'] == años_sorted[0]]['produccion'].mean()
            prod_final = df_trabajo[df_trabajo['año_numerico'] == años_sorted[-1]]['produccion'].mean()
            if prod_inicial > 0:
                cambio_total = ((prod_final - prod_inicial) / prod_inicial) * 100
            else:
                cambio_total = 0
        else:
            cambio_total = 0

        explanation = (
            f"🤖 PREDICCIÓN DE TENDENCIAS CON IA\n\n"
            f"📊 Datos usados: {total_datos:,} registros de producción agrícola\n"
            f"📅 Años analizados: {años_sorted[0]} - {años_sorted[-1]}\n"
            f"🌾 Producción total histórica: {produccion_total:,.0f} toneladas\n\n"
            f"🏆 MÉTODO UTILIZADO: {best_model_name}\n"
            f"   • Precisión del modelo: {r2:.2f} (más cerca de 1 = mejor)\n"
            f"   • Error promedio: {rmse:.0f} toneladas\n\n"
            f"🔮 PREDICCIONES PARA LOS PRÓXIMOS 5 AÑOS:\n"
            f"   • {años_futuros[0]}: {y_futuro[0]:,.0f} toneladas\n"
            f"   • {años_futuros[1]}: {y_futuro[1]:,.0f} toneladas\n"
            f"   • {años_futuros[2]}: {y_futuro[2]:,.0f} toneladas\n"
            f"   • {años_futuros[3]}: {y_futuro[3]:,.0f} toneladas\n"
            f"   • {años_futuros[4]}: {y_futuro[4]:,.0f} toneladas\n\n"
            f"📈 TENDENCIA GENERAL:\n"
            f"   • Cambio en el período estudiado: {cambio_total:+.1f}%\n"
            f"   • Dirección: {'📈 Producción subiendo' if cambio_total > 5 else '📉 Producción bajando' if cambio_total < -5 else '➡️ Producción estable'}\n\n"
            f"💡 ¿QUÉ SIGNIFICA ESTO?\n"
            f"   • La regresión lineal encontró la tendencia en tus datos históricos\n"
            f"   • Las predicciones siguen una línea recta basada en el patrón observado\n"
            f"   • Si la precisión es buena, puedes confiar en las estimaciones\n\n"
            f"📋 PARA QUÉ USARLO:\n"
            f"   • Planificar cuánta superficie sembrar\n"
            f"   • Decidir inversiones en agricultura\n"
            f"   • Prepararte para años buenos o malos"
        )

        messagebox.showinfo("Predicción Avanzada de Tendencias con IA",
                          f"Análisis completado y guardado en {output_file}\n\n{explanation}")

    def analisis_predictivo_nn(self):
        """Realiza un análisis predictivo utilizando una red neuronal simple para un cultivo seleccionado."""
        if not self._check_csv_loaded():
            return
        if 'sup_sembrada' not in self.df.columns or 'sup_cosechada' not in self.df.columns or 'rendimiento' not in self.df.columns or 'produccion' not in self.df.columns or 'cultivo' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener las columnas 'sup_sembrada', 'sup_cosechada', 'rendimiento', 'produccion' y 'cultivo'.")
            return

        # Seleccionar cultivo
        cultivos_disponibles = sorted(self.df['cultivo'].dropna().unique())
        if not cultivos_disponibles:
            messagebox.showwarning("Advertencia", "No se encontraron cultivos en los datos.")
            return

        selected_cultivo = self.ask_option("Seleccionar Cultivo", "Seleccione el cultivo para el análisis predictivo:", cultivos_disponibles)
        if not selected_cultivo:
            return

        # Filtrar datos por cultivo seleccionado
        df_cultivo = self.df[self.df['cultivo'] == selected_cultivo]

        # Limpiar datos eliminando filas con NaN
        df_clean = df_cultivo.dropna(subset=['sup_sembrada', 'sup_cosechada', 'rendimiento', 'produccion'])
        if len(df_clean) < 10:
            messagebox.showwarning("Advertencia", f"No hay suficientes datos válidos para el cultivo '{selected_cultivo}' después de eliminar valores NaN (mínimo 10 requeridos).")
            return

        # Preparar datos
        features = df_clean[['sup_sembrada', 'sup_cosechada', 'rendimiento']]
        target = df_clean['produccion']

        # Escalado de características con scalers separados
        scaler_features = MinMaxScaler()
        scaler_target = MinMaxScaler()
        features_scaled = scaler_features.fit_transform(features)
        target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1)).ravel()

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.2, random_state=42)

        # Construir modelo de red neuronal
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(3,)),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Compilar modelo
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Entrenar modelo
        model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

        # Evaluar modelo
        loss = model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"Pérdida en el conjunto de prueba: {loss}")

        # Predicciones
        predictions_scaled = model.predict(X_test, verbose=0)
        predictions_rescaled = scaler_target.inverse_transform(predictions_scaled).ravel()

        # Mostrar algunas predicciones
        logging.info(f"Algunas predicciones reescaladas: {predictions_rescaled[:5]}")
        logging.info(f"Valores reales correspondientes: {scaler_target.inverse_transform(y_test.reshape(-1, 1)).ravel()[:5]}")

        # Calcular métricas adicionales
        y_real = scaler_target.inverse_transform(y_test.reshape(-1, 1)).ravel()
        mse = mean_squared_error(y_real, predictions_rescaled)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_real - predictions_rescaled))
        r2 = r2_score(y_real, predictions_rescaled)

        # Estadísticas descriptivas del cultivo
        prod_media = df_clean['produccion'].mean()
        prod_std = df_clean['produccion'].std()
        prod_min = df_clean['produccion'].min()
        prod_max = df_clean['produccion'].max()

        # Comparación con modelo simple (predicción por la media)
        pred_media = np.full_like(y_real, prod_media)
        mse_media = mean_squared_error(y_real, pred_media)
        r2_media = r2_score(y_real, pred_media)

        # Crear gráficos de comparación
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(16, 6))

        # Gráfico 1: Real vs Predicho
        ax1.scatter(y_real, predictions_rescaled, alpha=0.6, color='blue', edgecolors='black')
        ax1.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], 'r--', linewidth=2, label='Línea ideal')
        ax1.set_xlabel('Producción Real (toneladas)')
        ax1.set_ylabel('Producción Predicha (toneladas)')
        ax1.set_title(f'Predicciones vs Realidad')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Gráfico 2: Distribución de errores (residuos)
        errores = y_real - predictions_rescaled
        ax2.scatter(predictions_rescaled, errores, alpha=0.6, color='green', edgecolors='black')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Sin error')
        ax2.set_xlabel('Producción Predicha (toneladas)')
        ax2.set_ylabel('Error (Real - Predicho)')
        ax2.set_title(f'Distribución de Errores\n(Error promedio: {errores.mean():.1f})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.suptitle(f"analisis_predictivo_nn - {selected_cultivo}", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
        plt.tight_layout()

        cultivo_filename = re.sub(r'[^\w\s-]', '', selected_cultivo).strip().replace(' ', '_')
        output_file = OUTPUT_DIR / f"red_neuronal_{cultivo_filename}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()

        explanation = (
            f"🧠 ANÁLISIS PREDICTIVO CON RED NEURONAL - {selected_cultivo.upper()}\n\n"
            f"Este análisis entrena una red neuronal artificial para predecir la producción del cultivo "
            f"'{selected_cultivo}' usando tres variables principales: superficie sembrada, superficie cosechada y rendimiento.\n\n"
            f"📊 DATOS UTILIZADOS:\n"
            f"   • Registros válidos analizados: {len(df_clean)}\n"
            f"   • Producción promedio histórica: {prod_media:.0f} toneladas\n"
            f"   • Producción mínima registrada: {prod_min:.0f} toneladas\n"
            f"   • Producción máxima registrada: {prod_max:.0f} toneladas\n"
            f"   • Desviación estándar: {prod_std:.0f} toneladas\n\n"
            f"🤖 CONFIGURACIÓN DEL MODELO:\n"
            f"   • Tipo: Red neuronal con 2 capas ocultas (10 y 8 neuronas)\n"
            f"   • Entradas: 3 variables (sup_sembrada, sup_cosechada, rendimiento)\n"
            f"   • Salida: 1 valor (producción predicha en toneladas)\n"
            f"   • Entrenamiento: 100 iteraciones con optimización automática\n"
            f"   • Validación: 20% de datos reservados para pruebas\n\n"
            f"📈 RESULTADOS DEL MODELO:\n"
            f"   • Error cuadrático medio (MSE): {mse:.0f} ton²\n"
            f"   • Error absoluto medio (MAE): {mae:.0f} toneladas\n"
            f"   • Raíz del error cuadrático medio (RMSE): {rmse:.0f} toneladas\n"
            f"   • Coeficiente de determinación (R²): {r2:.3f}\n\n"
            f"⚖️ COMPARACIÓN CON REFERENCIA:\n"
            f"   • Modelo de referencia (predicción constante = media): R² = {r2_media:.3f}\n"
            f"   • Mejora del modelo neuronal: {(r2 - r2_media)*100:.1f} puntos porcentuales\n\n"
            f"💡 INTERPRETACIÓN PRÁCTICA:\n"
            f"   • R² ≥ 0.8: Modelo muy preciso, predicciones confiables\n"
            f"   • R² 0.6-0.8: Modelo bueno, útil para estimaciones\n"
            f"   • R² 0.3-0.6: Modelo aceptable, considerar con precaución\n"
            f"   • R² < 0.3: Modelo deficiente, revisar calidad de datos\n\n"
            f"🔬 VENTAJAS DE LA RED NEURONAL:\n"
            f"   • Capacidad para modelar relaciones no lineales complejas\n"
            f"   • Procesamiento simultáneo de múltiples variables predictoras\n"
            f"   • Aprendizaje automático de patrones en los datos\n"
            f"   • Adaptabilidad a diferentes tipos de cultivos y condiciones\n\n"
            f"📋 APLICACIONES CONCRETAS:\n"
            f"   • Estimación de producción para planificación agrícola\n"
            f"   • Optimización de superficies de siembra\n"
            f"   • Evaluación de riesgos productivos\n"
            f"   • Soporte a decisiones de inversión en cultivos\n\n"
            f"📊 ANÁLISIS DE GRÁFICOS:\n"
            f"   • Gráfico izquierdo: Dispersión de valores reales vs predichos\n"
            f"   • Gráfico derecho: Distribución de errores residuales\n"
            f"   • Puntos cerca de la línea diagonal = mejores predicciones\n"
            f"   • Errores centrados en cero = modelo bien calibrado\n\n"
            f"⚠️ CONSIDERACIONES:\n"
            f"   • Las predicciones son válidas solo dentro del rango de datos históricos\n"
            f"   • Factores externos (clima, plagas) pueden afectar la precisión\n"
            f"   • Recomendado validar con datos futuros antes de decisiones críticas\n\n"
            f"📁 Archivo guardado: {output_file}"
        )
        # Crear ventana responsive con explicación completa y scroll
        ventana_nn = tk.Toplevel(self.root)
        ventana_nn.title("🧠 Resultados - Análisis Predictivo con Red Neuronal")
        ventana_nn.geometry("900x700")

        # Frame principal
        frame_principal = tk.Frame(ventana_nn)
        frame_principal.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título
        titulo_label = tk.Label(frame_principal, text=f"🧠 ANÁLISIS PREDICTIVO CON RED NEURONAL - {selected_cultivo.upper()}",
                               font=('Arial', 16, 'bold'), fg='#2563EB')
        titulo_label.pack(pady=(0, 20))

        # Text widget con scroll para la explicación completa
        frame_texto = tk.Frame(frame_principal)
        frame_texto.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(frame_texto)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        texto_nn = tk.Text(frame_texto, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                          font=('Arial', 11), padx=10, pady=10)
        texto_nn.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=texto_nn.yview)

        # Insertar explicación completa
        texto_nn.insert(tk.END, explanation)
        texto_nn.config(state=tk.DISABLED)  # Hacer solo lectura

        # Botón cerrar
        boton_cerrar = tk.Button(frame_principal, text="Cerrar",
                                command=ventana_nn.destroy,
                                font=('Arial', 12, 'bold'),
                                bg='#DC2626', fg='white', padx=20, pady=10)
        boton_cerrar.pack(pady=(20, 0))

        # Centrar ventana
        ventana_nn.transient(self.root)
        ventana_nn.grab_set()

    def geocodificar_direcciones(self):
        """Geocodifica direcciones con barra de progreso moderna y guarda las coordenadas en el DataFrame."""
        if not self._check_csv_loaded():
            return
        # Los nombres de columnas ya están normalizados a minúsculas en cargar_csv()
        if 'departamento' not in self.df.columns or 'provincia' not in self.df.columns or 'pais' not in self.df.columns:
            messagebox.showwarning("Advertencia", "Por favor, asegúrese de que el archivo CSV contenga las columnas 'departamento', 'provincia' y 'pais' (pueden estar en mayúsculas o con acentos).")
            return

        def geocode_with_retry(address, max_retries=3):
            for attempt in range(max_retries):
                try:
                    # Pausa más larga para respetar los límites del servicio
                    sleep(2)
                    location = geolocator.geocode(address, timeout=30)
                    return location
                except (GeocoderTimedOut, GeocoderServiceError) as e:
                    logging.warning(f"Error de geocodificación en intento {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        sleep(10)  # Espera más larga antes de reintentar
                        continue
                    else:
                        logging.error(f"Falló geocodificación después de {max_retries} intentos para: {address}")
                        return None
                except Exception as e:
                    logging.error(f"Error inesperado en geocodificación: {e}")
                    if attempt < max_retries - 1:
                        sleep(20)  # Espera aún más larga para errores de conexión
                        continue
                    else:
                        return None
            return None

        # Crear ventana de progreso moderna
        progress_window = tk.Toplevel(self.root)
        progress_window.title("🗺️ Progreso - Geocodificación")
        progress_window.geometry("550x250")
        progress_window.resizable(False, False)
        progress_window.grab_set()  # Hacer la ventana modal
        progress_window.configure(bg='#F8FAFC')

        # Centrar la ventana
        progress_window.transient(self.root)

        # Frame principal
        main_frame = tk.Frame(progress_window, bg='#F8FAFC')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=25)

        # Título moderno
        title_label = tk.Label(main_frame, text="🗺️ Geocodificando Direcciones",
                              font=('Arial', 16, 'bold'), fg='#2563EB', bg='#F8FAFC')
        title_label.pack(pady=(0, 20))

        # Información del progreso
        total_rows = len(self.df)
        info_label = tk.Label(main_frame, text=f"Procesando {total_rows} direcciones...",
                             font=('Arial', 12), fg='#64748B', bg='#F8FAFC')
        info_label.pack(pady=(0, 15))

        # Barra de progreso moderna
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(main_frame, variable=progress_var,
                                      maximum=100, length=450, mode='determinate')
        progress_bar.pack(pady=(0, 15))

        # Etiqueta de estado
        status_label = tk.Label(main_frame, text="⏳ Iniciando geocodificación...",
                               font=('Arial', 11), fg='#06B6D4', bg='#F8FAFC')
        status_label.pack(pady=(0, 10))

        # Etiqueta de progreso numérico
        progress_label = tk.Label(main_frame, text="0 / 0 (0%)",
                                 font=('Arial', 10), fg='#64748B', bg='#F8FAFC')
        progress_label.pack()

        latitudes = []
        longitudes = []
        addresses = []
        
        # Actualizar la ventana para mostrarla
        progress_window.update()

        # Usar contador manual para evitar problemas de tipo
        contador = 0
        for index, row in self.df.iterrows():
            contador += 1
            # Actualizar información de progreso
            current_progress = (contador / total_rows) * 100
            progress_var.set(current_progress)
            
            address = f"{row['departamento']}, {row['provincia']}, {row['pais']}"
            status_label.config(text=f"Procesando: {address[:50]}...")
            progress_label.config(text=f"{contador} / {total_rows} ({current_progress:.1f}%)")
            
            # Actualizar la interfaz
            progress_window.update()
            
            # Geocodificar la dirección
            location = geocode_with_retry(address)
            if location:
                latitudes.append(location.latitude)
                longitudes.append(location.longitude)
                addresses.append(location.address)
                status_label.config(text=f"✅ Encontrada: {location.address[:50]}...", fg="#10B981")
            else:
                latitudes.append(None)
                longitudes.append(None)
                addresses.append(None)
                status_label.config(text=f"❌ No encontrada: {address[:50]}...", fg="#EF4444")
            
            # Pequeña pausa para que se vea la actualización
            progress_window.update()
            sleep(0.1)

        # Finalizar progreso
        progress_var.set(100)
        status_label.config(text="Guardando resultados...", fg="blue")
        progress_label.config(text=f"{total_rows} / {total_rows} (100%)")
        progress_window.update()

        self.df['Latitude'] = latitudes
        self.df['Longitude'] = longitudes
        self.df['GeocodedAddress'] = addresses

        geocoded_file = OUTPUT_DIR / "geocodificado.csv"
        self.df.to_csv(geocoded_file, index=False)
        logging.info(f"Archivo CSV geocodificado guardado en {geocoded_file}")

        # Mostrar estadísticas finales
        successful_geocodes = sum(1 for lat in latitudes if lat is not None)
        failed_geocodes = total_rows - successful_geocodes
        
        status_label.config(text=f"🎉 Completado: {successful_geocodes} exitosas, {failed_geocodes} fallidas", fg="#10B981")
        progress_window.update()
        
        # Esperar un momento antes de cerrar
        sleep(1)
        progress_window.destroy()

        explanation = (
            "🗺️ GEOCODIFICACIÓN DE DIRECCIONES\n\n"
            "Este proceso convierte direcciones de texto en coordenadas GPS (latitud y longitud).\n\n"
            "🔍 ¿QUÉ HACE?\n"
            "   • Toma direcciones como 'Provincia X, País Y'\n"
            "   • Las convierte en números de ubicación\n"
            "   • Agrega columnas de Latitude y Longitude\n\n"
            "💡 ¿PARA QUÉ SIRVE?\n"
            "   • Crear mapas con tus datos\n"
            "   • Ver dónde están ubicadas las cosas\n"
            "   • Análisis geográfico de producción agrícola"
        )
        
        messagebox.showinfo("Geocodificación", 
                           f"Geocodificación completada.\n"
                           f"Direcciones procesadas: {total_rows}\n"
                           f"Geocodificaciones exitosas: {successful_geocodes}\n"
                           f"Geocodificaciones fallidas: {failed_geocodes}\n"
                           f"Archivo guardado en: {geocoded_file}\n\n{explanation}")

    def generar_mapa(self):
        """Genera un mapa con las direcciones geocodificadas."""
        if not self._check_csv_loaded():
            return
        if 'Latitude' not in self.df.columns or 'Longitude' not in self.df.columns:
            messagebox.showwarning("Advertencia", "Por favor, geocodifique las direcciones primero.")
            return

        centro = [self.df['Latitude'].mean(), self.df['Longitude'].mean()]
        mapa = folium.Map(location=centro, zoom_start=6)

        for _, row in self.df.iterrows():
            if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=f"Provincia: {row['provincia']}, Departamento: {row['departamento']}, Cultivo: {row['cultivo']}",
                ).add_to(mapa)

        mapa_file = OUTPUT_DIR / "mapa_geoespacial.html"
        mapa.save(mapa_file)
        logging.info(f"Mapa geoespacial guardado en {mapa_file}")

        webbrowser.open(mapa_file.resolve().as_uri())

        explanation = (
            "🗺️ MAPA GEOESPACIAL\n\n"
            "Este análisis crea un mapa interactivo con puntos en las ubicaciones de tus datos.\n\n"
            "🔍 ¿QUÉ VERÁS?\n"
            "   • Puntos en el mapa = ubicaciones de tus datos\n"
            "   • Al hacer clic en un punto = Provincia, Departamento y Cultivo\n\n"
            "💡 ¿PARA QUÉ SIRVE?\n"
            "   • Ver dónde se produce más agricultura\n"
            "   • Identificar patrones geográficos\n"
            "   • Planificar distribución de recursos"
        )
        messagebox.showinfo("Generar Mapa", f"Mapa generado exitosamente.\n\n{explanation}")

    def mapa_distribucion_cultivos(self):
        """Genera un mapa del mundo mostrando la distribución de cultivos con colores diferenciados."""
        if not self._check_csv_loaded():
            return
        if 'Latitude' not in self.df.columns or 'Longitude' not in self.df.columns or 'cultivo' not in self.df.columns:
            messagebox.showwarning("Advertencia", "Por favor, geocodifique las direcciones primero y asegúrese de tener la columna 'cultivo'.")
            return

        # Filtrar datos con coordenadas válidas
        df_mapa = self.df.dropna(subset=['Latitude', 'Longitude', 'cultivo'])

        if df_mapa.empty:
            messagebox.showwarning("Advertencia", "No hay datos válidos con coordenadas y cultivos para mostrar en el mapa.")
            return

        # Crear mapa centrado en el mundo
        mapa = folium.Map(location=[0, 0], zoom_start=2)

        # Obtener cultivos únicos y asignar colores
        cultivos_unicos = df_mapa['cultivo'].unique()
        colores = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen',
                   'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']

        color_dict = {}
        for i, cultivo in enumerate(cultivos_unicos):
            color_dict[cultivo] = colores[i % len(colores)]

        # Agregar marcadores para cada punto de datos
        for _, row in df_mapa.iterrows():
            cultivo = row['cultivo']
            color = color_dict[cultivo]
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                popup=f"Cultivo: {cultivo}<br>Provincia: {row.get('provincia', 'N/A')}<br>Departamento: {row.get('departamento', 'N/A')}",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7
            ).add_to(mapa)

        # Agregar leyenda
        legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 200px; height: auto;
                    background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
                    padding: 10px">
        <p><b>Leyenda de Cultivos</b></p>
        '''
        for cultivo, color in color_dict.items():
            legend_html += f'<p><span style="color:{color};">&#9679;</span> {cultivo}</p>'
        legend_html += '</div>'

        mapa.get_root().html.add_child(folium.Element(legend_html))

        mapa_file = OUTPUT_DIR / "mapa_distribucion_cultivos.html"
        mapa.save(mapa_file)
        logging.info(f"Mapa de distribución de cultivos guardado en {mapa_file}")

        webbrowser.open(mapa_file.resolve().as_uri())

        explanation = (
            "🌍 MAPA DE DISTRIBUCIÓN DE CULTIVOS\n\n"
            "Este mapa muestra la distribución mundial de tus cultivos agrícolas, "
            "con cada tipo de cultivo representado por un color diferente.\n\n"
            "🔍 ¿QUÉ VERÁS?\n"
            "   • Puntos coloreados = ubicaciones de cultivos\n"
            "   • Cada color representa un tipo de cultivo diferente\n"
            "   • Leyenda en la esquina inferior izquierda\n\n"
            "💡 ¿PARA QUÉ SIRVE?\n"
            "   • Ver dónde se cultivan diferentes productos\n"
            "   • Identificar patrones globales de agricultura\n"
            "   • Analizar diversidad agrícola por región"
        )
        messagebox.showinfo("Mapa de Distribución de Cultivos", f"Mapa generado exitosamente.\n\n{explanation}")

    def produccion_top_cultivos(self):
        """Genera un gráfico de líneas para los 4 principales cultivos por producción total."""
        if not self._check_csv_loaded():
            return
        if 'cultivo' not in self.df.columns or 'campana' not in self.df.columns or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener las columnas 'cultivo', 'campana' y 'produccion'.")
            return

        # Agrupar los datos por cultivo y campana, y sumar la producción
        grouped_data = self.df.groupby(['cultivo', 'campana'])['produccion'].sum().reset_index()

        # Obtener los 4 principales cultivos por producción total
        top_cultivos = grouped_data.groupby('cultivo')['produccion'].sum().nlargest(4).index

        # Filtrar los datos para incluir solo los 4 cultivos principales
        filtered_data = grouped_data[grouped_data['cultivo'].isin(top_cultivos)]

        # Crear un gráfico de líneas que muestre la producción por campana para los 4 cultivos principales
        plt.figure(figsize=(12, 8))
        for cultivo in top_cultivos:
            cultivo_data = filtered_data[filtered_data['cultivo'] == cultivo]
            plt.plot(cultivo_data['campana'], cultivo_data['produccion'], marker='o', label=cultivo)

        plt.title('Producción de los 4 principales cultivos por campaña')
        plt.xlabel('Campaña')
        plt.ylabel('Producción (en toneladas)')
        plt.xticks(rotation=45)
        plt.legend(title='Cultivo')
        plt.grid(True)
        plt.suptitle("produccion_top_cultivos", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
        plt.tight_layout()

        output_file = OUTPUT_DIR / "produccion_top_cultivos.png"
        plt.savefig(output_file)
        plt.show()
        logging.info(f"Gráfica de producción de los 4 principales cultivos guardada en {output_file}")

        explanation = (
            "🌱 PRODUCCIÓN DE LOS 4 CULTIVOS PRINCIPALES\n\n"
            "Esta gráfica muestra cómo ha cambiado la producción de los cultivos más importantes con el tiempo.\n\n"
            "🔍 ¿QUÉ VER?\n"
            "   • Líneas que suben = producción aumentando\n"
            "   • Líneas que bajan = producción disminuyendo\n"
            "   • Cada color representa un cultivo diferente\n\n"
            "💡 ¿PARA QUÉ SIRVE?\n"
            "   • Saber qué cultivos están de moda\n"
            "   • Planificar qué sembrar en el futuro\n"
            "   • Tomar decisiones de inversión"
        )
        messagebox.showinfo("Producción Top Cultivos", f"Gráfica guardada en {output_file}\n\n{explanation}")


    def mostrar_dialogo_informes(self):
        """Muestra un cuadro de diálogo para seleccionar y generar informes."""
        informes = ["Producción Total por Provincia", "Correlación Sup. Sembrada-Sup. Cosechada", "Sumar Columnas", 
                    "Análisis Temporal", "Análisis de Correlación", "Modelos Predictivos", 
                    "Clasificación de Cultivos", "Análisis de Riesgos", "Evolución de Cultivos por Campaña", 
                    "Tendencias de Producción por Cultivo", "Predicción de Tendencias con IA",
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
            "Evolución de Cultivos por Campaña": "evolucion_cultivos_por_campana",
            "Tendencias de Producción por Cultivo": "tendencias_produccion_por_cultivo",
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

    def acerca_de(self):
        """Muestra información sobre la aplicación."""
        info = (
            "🌾 APLICACIÓN DE ANÁLISIS AGRÍCOLA CON IA\n\n"
            "Versión: 2025.1.0\n"
            "Desarrollado para análisis de datos agrícolas.\n\n"
            "Características principales:\n"
            "• Análisis estadístico integral de variables numéricas\n"
            "• Análisis temporal de producción agrícola\n"
            "• Análisis de correlación entre variables\n"
            "• Modelos predictivos con IA\n"
            "• Clasificación y análisis de cultivos\n"
            "• Análisis de riesgos por provincia\n"
            "• Geocodificación y mapas interactivos\n\n"
            "Tecnologías utilizadas:\n"
            "• Python con Tkinter para interfaz gráfica\n"
            "• Pandas para manipulación de datos\n"
            "• Matplotlib y Seaborn para visualizaciones\n"
            "• Scikit-learn para modelos de IA\n"
            "• Folium para mapas interactivos\n"
            "• TensorFlow para redes neuronales\n\n"
            "Desarrollado para tesis académica en agricultura."
        )
        messagebox.showinfo("Acerca de", info)

    def manual_usuario(self):
        """Muestra el manual de usuario de la aplicación en una ventana desplazable."""
        manual = (
            "📖 MANUAL DE USUARIO - ANÁLISIS AGRÍCOLA CON IA\n\n"
            "PASO 1: CARGAR DATOS\n"
            "• Selecciona 'Archivo' > 'Cargar CSV'\n"
            "• Elige un archivo CSV con datos agrícolas\n"
            "• Columnas recomendadas: campana, cultivo, provincia, produccion, sup_sembrada, sup_cosechada, rendimiento\n\n"
            "PASO 2: ANÁLISIS DISPONIBLES\n\n"
            "📊 SUMAR COLUMNAS:\n"
            "• Analiza estadísticas de todas las variables numéricas\n"
            "• Genera gráficos de totales, promedios y coeficiente de variación\n\n"
            "📈 ANÁLISIS TEMPORAL:\n"
            "• Muestra evolución de producción por años\n"
            "• Requiere columna 'campana'\n\n"
            "🔗 ANÁLISIS DE CORRELACIÓN:\n"
            "• Identifica relaciones entre variables\n"
            "• Muestra matriz de correlación y gráficos\n\n"
            "🤖 PREDICCIÓN DE TENDENCIAS CON IA:\n"
            "• Predice producción futura usando regresión lineal\n"
            "• Requiere columnas 'campana' y 'produccion'\n\n"
            "📈 MODELOS PREDICTIVOS:\n"
            "• Modelo de regresión lineal para superficie vs producción\n"
            "• Requiere 'sup_sembrada' y 'produccion'\n\n"
            "🌱 CLASIFICACIÓN DE CULTIVOS:\n"
            "• Analiza distribución y características de cultivos\n"
            "• Requiere columna 'cultivo'\n\n"
            "⚠️ ANÁLISIS DE RIESGOS:\n"
            "• Clasifica zonas por nivel de riesgo de producción\n"
            "• Opciones: Global, Por Provincia, Por Cultivo\n\n"
            "🗺️ GEOCODIFICACIÓN:\n"
            "• Convierte direcciones en coordenadas GPS\n"
            "• Requiere 'departamento', 'provincia', 'pais'\n\n"
            "🗺️ GENERAR MAPA:\n"
            "• Crea mapa interactivo con ubicaciones\n"
            "• Requiere coordenadas geocodificadas\n\n"
            "🌍 MAPA DE DISTRIBUCIÓN DE CULTIVOS:\n"
            "• Muestra cultivos en mapa mundial\n"
            "• Requiere coordenadas y columna 'cultivo'\n\n"
            "PASO 3: INTERPRETAR RESULTADOS\n\n"
            "• Cada análisis genera gráficos guardados en carpeta 'output/'\n"
            "• Los gráficos incluyen explicaciones detalladas\n"
            "• Mapas se abren automáticamente en navegador\n\n"
            "REQUISITOS DEL CSV:\n"
            "• Formato: CSV separado por comas\n"
            "• Codificación: UTF-8\n"
            "• Columnas: Nombres en español o inglés (se normalizan)\n"
            "• Datos: Valores numéricos para análisis estadístico\n\n"
            "TIPS PARA MEJORES RESULTADOS:\n"
            "• Asegúrate de que los datos estén limpios (sin valores nulos en columnas clave)\n"
            "• Usa al menos 10 registros para análisis predictivos\n"
            "• Para mapas, geocodifica primero las direcciones\n"
            "• Los gráficos se guardan automáticamente en alta resolución"
        )

        # Crear ventana con explicación completa y scroll
        ventana_manual = tk.Toplevel(self.root)
        ventana_manual.title("📖 Manual de Usuario - Análisis Agrícola con IA")
        ventana_manual.geometry("900x700")

        # Frame principal
        frame_principal = tk.Frame(ventana_manual)
        frame_principal.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título
        titulo_label = tk.Label(frame_principal, text="📖 MANUAL DE USUARIO DETALLADO",
                               font=('Arial', 16, 'bold'), fg='#2563EB')
        titulo_label.pack(pady=(0, 20))

        # Text widget con scroll para el manual completo
        frame_texto = tk.Frame(frame_principal)
        frame_texto.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(frame_texto)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        texto_manual = tk.Text(frame_texto, wrap=tk.WORD, yscrollcommand=scrollbar.set,
                              font=('Arial', 11), padx=10, pady=10)
        texto_manual.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=texto_manual.yview)

        # Insertar manual completo
        texto_manual.insert(tk.END, manual)
        texto_manual.config(state=tk.DISABLED)  # Hacer solo lectura

        # Botón cerrar
        boton_cerrar = tk.Button(frame_principal, text="Cerrar",
                                command=ventana_manual.destroy,
                                font=('Arial', 12, 'bold'),
                                bg='#DC2626', fg='white', padx=20, pady=10)
        boton_cerrar.pack(pady=(20, 0))

        # Centrar ventana
        ventana_manual.transient(self.root)
        ventana_manual.grab_set()

    @staticmethod
    def safe_file_name(name):
        """Devuelve un nombre de archivo seguro para usar en el sistema de archivos."""
        return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
 
 
 
if __name__ == "__main__":
    app = DataAnalyzer()
    app.root.mainloop()
