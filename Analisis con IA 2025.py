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

# Configuración del registro
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directorio de salida
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        """Realiza un análisis estadístico integral de las variables numéricas del dataset agrícola."""
        if self.df.empty:
            messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return

        # Obtener columnas numéricas
        numeric_cols = self.df.select_dtypes(include=[float, int]).columns.tolist()
        
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
        
        # Crear visualización mejorada con múltiples subgráficos
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Gráfico 1: Totales por variable (suma)
        suma_columnas.plot(kind='bar', ax=ax1, color='lightblue', edgecolor='navy')
        ax1.set_title('Totales Acumulados por Variable')
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
        ax2.set_title('Valores Promedio por Variable')
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
        ax3.set_title('Coeficiente de Variación por Variable (%)')
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
        
        # Gráfico 4: Comparación Min-Max-Promedio
        variables_principales = suma_columnas.nlargest(6).index  # Top 6 variables
        df_principales = df_numeric[variables_principales]
        
        x_pos = np.arange(len(variables_principales))
        width = 0.25
        
        mins = df_principales.min()
        maxs = df_principales.max()
        means = df_principales.mean()
        
        ax4.bar(x_pos - width, mins, width, label='Mínimo', color='lightcoral', alpha=0.8)
        ax4.bar(x_pos, means, width, label='Promedio', color='lightskyblue', alpha=0.8)
        ax4.bar(x_pos + width, maxs, width, label='Máximo', color='lightgreen', alpha=0.8)
        
        ax4.set_title('Comparación Min-Promedio-Max (Top 6 Variables)')
        ax4.set_xlabel('Variables Principales')
        ax4.set_ylabel('Valores')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(variables_principales, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle("sumar_columnas", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
        plt.tight_layout()
        
        # Guardar gráfico
        output_file = OUTPUT_DIR / "analisis_estadistico_integral.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        logging.info(f"Análisis estadístico integral guardado en {output_file}")
        
        # Análisis de correlaciones entre variables principales
        correlaciones_importantes = []
        if len(variables_principales) > 1:
            corr_matrix = df_principales.corr()
            # Encontrar correlaciones fuertes (>0.7 o <-0.7)
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        var1 = corr_matrix.columns[i]
                        var2 = corr_matrix.columns[j]
                        correlaciones_importantes.append(f"{var1} ↔ {var2}: {corr_val:.3f}")
        
        # Identificar outliers usando el método IQR
        outliers_info = []
        for col in variables_principales:
            Q1 = df_numeric[col].quantile(0.25)
            Q3 = df_numeric[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df_numeric[(df_numeric[col] < lower_bound) | (df_numeric[col] > upper_bound)][col]
            if len(outliers) > 0:
                outliers_info.append(f"{col}: {len(outliers)} valores atípicos ({len(outliers)/len(df_numeric)*100:.1f}%)")
        
        # Crear reporte detallado
        correlaciones_texto = "\n".join(correlaciones_importantes[:5]) if correlaciones_importantes else "No se encontraron correlaciones fuertes (>0.7)"
        outliers_texto = "\n".join(outliers_info[:5]) if outliers_info else "No se detectaron valores atípicos significativos"
        
        explanation = (
            f"📊 ANÁLISIS ESTADÍSTICO INTEGRAL DE VARIABLES NUMÉRICAS\n\n"
            f"🔍 Datos analizados: {len(df_numeric):,} registros válidos\n"
            f"📈 Variables numéricas: {len(numeric_cols)} columnas\n\n"
            f"🏆 VARIABLES DESTACADAS:\n"
            f"   • Mayor volumen total: {variable_mayor_suma} ({suma_columnas[variable_mayor_suma]:,.0f})\n"
            f"   • Más variable: {variable_mayor_variabilidad} (CV: {coef_variacion[variable_mayor_variabilidad]:.1f}%)\n"
            f"   • Más estable: {variable_mas_estable} (CV: {coef_variacion[variable_mas_estable]:.1f}%)\n\n"
            f"📊 ESTADÍSTICAS CLAVE:\n"
            f"   • Promedio general: {df_numeric.mean().mean():,.1f}\n"
            f"   • Desviación estándar promedio: {df_numeric.std().mean():,.1f}\n"
            f"   • Coeficiente de variación promedio: {coef_variacion.mean():.1f}%\n\n"
            f"🔗 CORRELACIONES IMPORTANTES:\n{correlaciones_texto}\n\n"
            f"⚠️ VALORES ATÍPICOS DETECTADOS:\n{outliers_texto}\n\n"
            f"💡 INTERPRETACIÓN PARA TESIS:\n"
            f"   • Las variables con mayor volumen indican los aspectos más significativos del dataset\n"
            f"   • El coeficiente de variación revela la estabilidad/volatilidad de cada variable\n"
            f"   • Las correlaciones fuertes sugieren relaciones causales o dependencias\n"
            f"   • Los valores atípicos pueden indicar casos especiales o errores de medición\n\n"
            f"📋 APLICACIONES PRÁCTICAS:\n"
            f"   • Identificación de variables clave para modelos predictivos\n"
            f"   • Detección de inconsistencias en los datos\n"
            f"   • Priorización de variables para análisis posteriores\n"
            f"   • Fundamentación estadística para decisiones metodológicas"
        )
        
        messagebox.showinfo("Análisis Estadístico Integral", f"Análisis completado y guardado en {output_file}\n\n{explanation}")

    def analisis_temporal(self):
        """Genera un análisis temporal de la producción."""
        if self.df.empty or 'campaña' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener la columna 'campaña'.")
            return

        if 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener la columna 'produccion'.")
            return

        # Integración del nuevo análisis temporal
        self.df['campaña'] = self.df['campaña'].astype(str).str.split('/').str[0].astype(int)
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

        plt.suptitle("analisis_temporal", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
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
        plt.suptitle("analisis_correlacion", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
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

        # Convertir la columna campaña a string para evitar errores
        self.df['campaña'] = self.df['campaña'].astype(str)
        
        campañas = self.df['campaña'].unique()
        if len(campañas) == 0:
            messagebox.showwarning("Advertencia", "No se encontraron campañas en el archivo CSV.")
            return

        campañas_limpias = [str(campaña).strip() for campaña in campañas if pd.notna(campaña)]

        selected_campaña = self.ask_option("Seleccionar Campaña", "Seleccione la campaña:", campañas_limpias)
        if not selected_campaña:
            return

        # Filtrar usando comparación directa en lugar de .str.strip()
        df_campaña = self.df[self.df['campaña'].astype(str).str.strip() == selected_campaña]
        
        if df_campaña.empty:
            messagebox.showwarning("Advertencia", "No se encontraron datos para la campaña seleccionada.")
            return
            
        produccion_por_provincia = df_campaña.groupby('provincia')['produccion'].sum().sort_values(ascending=False)

        if produccion_por_provincia.empty:
            messagebox.showwarning("Advertencia", "No se encontraron datos de producción para la campaña seleccionada.")
            return

        title = f"Producción Total por Provincia - Campaña {selected_campaña}"
        output_file = OUTPUT_DIR / f"produccion_por_provincia_{self.safe_file_name(selected_campaña)}.png"
        Visualization.plot_bar_chart(produccion_por_provincia, title, "Provincias", "Producción [Tn]", output_file, "produccion_total_por_provincia")

        explanation = (
            "Este informe muestra la producción total de cultivos por provincia para la campaña seleccionada. "
            "Permite identificar qué provincias tienen mayor y menor producción, lo cual puede ayudar en la toma de decisiones "
            "para la distribución de recursos y planificación agrícola."
        )
        messagebox.showinfo("Producción Total por Provincia", f"Gráfica guardada en {output_file}\n\n{explanation}")

    def evolucion_cultivos_por_campaña(self):
        """Genera un gráfico de la evolución de los cultivos por campaña con nombres limpios y legibles."""
        if self.df.empty or 'campaña' not in self.df.columns or 'cultivo' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener las columnas 'campaña' y 'cultivo'.")
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

        # Procesar fechas de campaña de manera más robusta
        try:
            # Intentar diferentes formatos de fecha
            if df_trabajo['campaña'].dtype == 'object':
                # Si es texto, intentar extraer el año
                df_trabajo['año'] = df_trabajo['campaña'].astype(str).str.extract(r'(\d{4})').astype(float)
            else:
                # Si es numérico, usar directamente
                df_trabajo['año'] = pd.to_numeric(df_trabajo['campaña'], errors='coerce')
            
            # Filtrar años válidos
            df_trabajo = df_trabajo.dropna(subset=['año'])
            df_trabajo['año'] = df_trabajo['año'].astype(int)
            
        except Exception as e:
            logging.error(f"Error procesando fechas de campaña: {e}")
            messagebox.showerror("Error", "No se pudieron procesar las fechas de campaña correctamente.")
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

        # Crear visualización mejorada
        plt.figure(figsize=(14, 10))
        
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

        plt.suptitle("evolucion_cultivos_por_campaña", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
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
            f"📊 EVOLUCIÓN DEL CULTIVO: {cultivo_seleccionado.upper()}\n\n"
            f"📅 Período analizado: {año_inicial} - {año_final} ({años_analizados} años)\n"
            f"📈 Variables analizadas: {', '.join([col.replace('_', ' ').title() for col in columnas_presentes])}\n\n"
            f"📊 TENDENCIAS IDENTIFICADAS:\n{tendencias_texto}\n"
            f"💡 INTERPRETACIÓN:\n"
            f"   • Este gráfico muestra la evolución temporal del cultivo {cultivo_seleccionado}\n"
            f"   • Permite identificar patrones de crecimiento, declive o estabilidad\n"
            f"   • Los valores en cada punto muestran las cantidades exactas por año\n"
            f"   • Útil para planificación agrícola y toma de decisiones estratégicas\n\n"
            f"📋 APLICACIONES PRÁCTICAS:\n"
            f"   • Identificación de años de alta/baja productividad\n"
            f"   • Análisis de impacto de factores climáticos o económicos\n"
            f"   • Planificación de siembra basada en tendencias históricas\n"
            f"   • Evaluación de la viabilidad del cultivo a largo plazo"
        )
        
        messagebox.showinfo("Evolución de Cultivo por Campaña", f"Gráfica guardada en {evolucion_file}\n\n{explanation}")

    def tendencias_produccion_por_cultivo(self):
        """Genera un gráfico de tendencias de producción por cultivo y campaña mejorado."""
        if self.df.empty or 'campaña' not in self.df.columns or 'cultivo' not in self.df.columns or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El archivo CSV debe contener las columnas 'campaña', 'cultivo' y 'produccion'.")
            return

        # Filtrar datos válidos
        df_valid = self.df.dropna(subset=['campaña', 'cultivo', 'produccion']).copy()
        
        if len(df_valid) < 10:
            messagebox.showwarning("Advertencia", "No hay suficientes datos válidos para el análisis de tendencias.")
            return

        # Agrupar por cultivo y campaña, sumando la producción
        df_grouped = df_valid.groupby(['cultivo', 'campaña'])['produccion'].sum().reset_index()
        
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
            ax1.plot(cultivo_data['campaña'], cultivo_data['produccion'],
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
                ax2.plot(cultivo_data['campaña'], cultivo_data['produccion'],
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
        evolucion_promedio = df_grouped.groupby('campaña')['produccion'].mean()
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
            f"📊 ANÁLISIS DE TENDENCIAS DE PRODUCCIÓN POR CULTIVO\n\n"
            f"🔍 Datos analizados: {len(df_valid):,} registros de producción\n"
            f"🌱 Cultivos analizados: {len(df_valid['cultivo'].unique())}\n"
            f"📅 Campañas analizadas: {len(df_valid['campaña'].unique())}\n\n"
            f"🏆 Top 3 Cultivos por Producción Total:\n"
            f"   1. {produccion_total_por_cultivo.index[0]}: {produccion_total_por_cultivo.iloc[0]:,.0f} toneladas\n"
            f"   2. {produccion_total_por_cultivo.index[1]}: {produccion_total_por_cultivo.iloc[1]:,.0f} toneladas\n"
            f"   3. {produccion_total_por_cultivo.index[2]}: {produccion_total_por_cultivo.iloc[2]:,.0f} toneladas\n\n"
            f"📈 Análisis de Estabilidad:\n"
            f"   🟢 Cultivo más estable: {cultivo_mas_estable if cultivo_mas_estable else 'No disponible'}\n"
            f"   🔴 Cultivo más variable: {cultivo_mas_variable if cultivo_mas_variable else 'No disponible'}\n\n"
            f"💡 ¿Qué muestran estos gráficos?\n"
            f"   • Evolución temporal de cada cultivo principal\n"
            f"   • Comparación de volúmenes de producción\n"
            f"   • Tendencia general del sector agrícola\n"
            f"   • Identificación de cultivos estables vs volátiles\n\n"
            f"📋 Utilidad práctica:\n"
            f"   • Planificación basada en tendencias históricas\n"
            f"   • Identificación de cultivos en crecimiento/declive\n"
            f"   • Gestión de riesgos por variabilidad\n"
            f"   • Diversificación estratégica de cultivos"
        )
        
        messagebox.showinfo("Tendencias de Producción por Cultivo", f"Gráfica guardada en {tendencias_file}\n\n{explanation}")

    def modelos_predictivos(self):
        """Entrena y evalúa un modelo de regresión lineal."""
        if self.df.empty or 'sup_sembrada' not in self.df.columns or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener 'sup_sembrada' y 'produccion'.")
            return

        # Limpiar datos eliminando filas con NaN en las columnas relevantes
        self.df = self.df.dropna(subset=['sup_sembrada', 'produccion'])

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
        """Analiza y clasifica cultivos según características de producción."""
        columnas_requeridas = ['cultivo']
        columnas_opcionales = ['sup_sembrada', 'sup_cosechada', 'produccion', 'rendimiento', 'provincia']
        
        if self.df.empty or 'cultivo' not in self.df.columns:
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
        """Realiza un análisis de riesgos agrícolas identificando zonas de alta, media y baja producción por provincia y campaña."""
        columnas_requeridas = ['produccion']
        columnas_opcionales = ['provincia', 'campaña', 'departamento']
        
        # Verificar columnas requeridas
        if self.df.empty or not all(col in self.df.columns for col in columnas_requeridas):
            messagebox.showwarning("Advertencia", "El DataFrame debe contener la columna 'produccion'.")
            return

        # Filtrar filas con datos válidos en 'produccion'
        df_valid = self.df[self.df['produccion'].notna() & (self.df['produccion'] > 0)].copy()

        if len(df_valid) < 10:
            messagebox.showwarning("Advertencia", "No hay suficientes datos válidos para realizar el análisis de riesgos.")
            return

        # Limitar a una muestra para evitar consumo excesivo de RAM
        if len(df_valid) > 5000:
            df_valid = df_valid.sample(n=5000, random_state=42)
            logging.info("Muestra limitada a 5000 filas para análisis de riesgos.")

        # Obtener información temporal si está disponible
        campañas_info = ""
        if 'campaña' in df_valid.columns:
            campañas_unicas = sorted(df_valid['campaña'].dropna().unique())
            if len(campañas_unicas) > 0:
                primera_campaña = campañas_unicas[0]
                ultima_campaña = campañas_unicas[-1]
                total_campañas = len(campañas_unicas)
                campañas_info = f"📅 Período analizado: {primera_campaña} - {ultima_campaña} ({total_campañas} campañas)\n"

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

        # Análisis por provincia si está disponible
        zonas_alto_riesgo = []
        zonas_medio_riesgo = []
        zonas_bajo_riesgo = []
        
        if 'provincia' in df_valid.columns:
            # Agrupar por provincia y calcular producción promedio
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

        # Gráfico 3: Producción por provincia si está disponible
        if 'provincia' in df_valid.columns and len(produccion_por_provincia) <= 15:
            produccion_por_provincia_sorted = produccion_por_provincia.sort_values('Produccion_Promedio')
            colores_provincias = ['red' if x == 'Alto Riesgo' else 'orange' if x == 'Riesgo Medio' else 'green'
                                for x in produccion_por_provincia_sorted['Nivel_Riesgo_Predominante']]
            
            bars = ax3.barh(produccion_por_provincia_sorted['provincia'],
                           produccion_por_provincia_sorted['Produccion_Promedio'],
                           color=colores_provincias)
            ax3.set_xlabel('Producción Promedio (toneladas)')
            ax3.set_ylabel('Provincia')
            ax3.set_title('Producción Promedio por Provincia')
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

        plt.suptitle("analisis_riesgos", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
        plt.tight_layout()
        
        # Guardar gráfico
        riesgo_file = OUTPUT_DIR / "analisis_riesgos_agricola.png"
        plt.savefig(riesgo_file, dpi=300, bbox_inches='tight')
        plt.show()
        logging.info(f"Análisis de riesgos guardado en {riesgo_file}")

        # Asignar clasificación al DataFrame principal
        self.df.loc[df_valid.index, 'Nivel_Riesgo'] = df_valid['Nivel_Riesgo']

        # Crear reporte detallado
        porcentaje_alto = (conteo_riesgos.get('Alto Riesgo', 0) / len(df_valid)) * 100
        porcentaje_medio = (conteo_riesgos.get('Riesgo Medio', 0) / len(df_valid)) * 100
        porcentaje_bajo = (conteo_riesgos.get('Bajo Riesgo', 0) / len(df_valid)) * 100

        # Construir información de zonas
        zonas_info = ""
        if zonas_alto_riesgo or zonas_medio_riesgo or zonas_bajo_riesgo:
            zonas_info += "\n🗺️ ZONAS IDENTIFICADAS:\n"
            if zonas_alto_riesgo:
                zonas_info += f"   🔴 ALTO RIESGO: {', '.join(zonas_alto_riesgo[:5])}"
                if len(zonas_alto_riesgo) > 5:
                    zonas_info += f" y {len(zonas_alto_riesgo)-5} más"
                zonas_info += "\n"
            if zonas_medio_riesgo:
                zonas_info += f"   🟡 RIESGO MEDIO: {', '.join(zonas_medio_riesgo[:5])}"
                if len(zonas_medio_riesgo) > 5:
                    zonas_info += f" y {len(zonas_medio_riesgo)-5} más"
                zonas_info += "\n"
            if zonas_bajo_riesgo:
                zonas_info += f"   🟢 BAJO RIESGO: {', '.join(zonas_bajo_riesgo[:5])}"
                if len(zonas_bajo_riesgo) > 5:
                    zonas_info += f" y {len(zonas_bajo_riesgo)-5} más"
                zonas_info += "\n"

        explanation = (
            f"📊 ANÁLISIS DE RIESGOS AGRÍCOLAS\n\n"
            f"{campañas_info}"
            f"🔍 Datos analizados: {len(df_valid):,} registros de producción\n\n"
            f"📈 Estadísticas de Producción:\n"
            f"   • Producción mínima: {min_produccion:,.0f} toneladas\n"
            f"   • Producción máxima: {max_produccion:,.0f} toneladas\n"
            f"   • Producción promedio: {media_produccion:,.0f} toneladas\n\n"
            f"⚠️ Clasificación de Riesgos:\n"
            f"   🔴 ALTO RIESGO (≤{percentil_33:.0f} ton): {conteo_riesgos.get('Alto Riesgo', 0)} casos ({porcentaje_alto:.1f}%)\n"
            f"   🟡 RIESGO MEDIO ({percentil_33:.0f}-{percentil_66:.0f} ton): {conteo_riesgos.get('Riesgo Medio', 0)} casos ({porcentaje_medio:.1f}%)\n"
            f"   🟢 BAJO RIESGO (>{percentil_66:.0f} ton): {conteo_riesgos.get('Bajo Riesgo', 0)} casos ({porcentaje_bajo:.1f}%)\n"
            f"{zonas_info}\n"
            f"💡 Interpretación:\n"
            f"   • Las zonas de ALTO RIESGO requieren atención inmediata\n"
            f"   • Las zonas de RIESGO MEDIO necesitan monitoreo\n"
            f"   • Las zonas de BAJO RIESGO son las más productivas\n\n"
            f"📋 Recomendaciones:\n"
            f"   • Investigar causas en zonas de alto riesgo (clima, suelo, plagas)\n"
            f"   • Implementar mejores prácticas en zonas de riesgo medio\n"
            f"   • Replicar estrategias exitosas de zonas de bajo riesgo"
        )
        
        messagebox.showinfo("Análisis de Riesgos Agrícolas", explanation)

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
        """Realiza predicción avanzada de tendencias agrícolas usando múltiples algoritmos de IA con optimización de hiperparámetros."""
        if self.df.empty or 'campaña' not in self.df.columns or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener las columnas 'campaña' y 'produccion'.")
            return

        # Preparar datos
        df_trabajo = self.df.dropna(subset=['campaña', 'produccion']).copy()
        if len(df_trabajo) < 10:
            messagebox.showwarning("Advertencia", "Se necesitan al menos 10 registros para el análisis predictivo.")
            return

        # Convertir campaña a valores numéricos para el análisis (manejar formato "2023/2024")
        try:
            # Intentar convertir campañas al formato usado en otras funciones
            df_trabajo['año_numerico'] = df_trabajo['campaña'].astype(str).str.split('/').str[0].astype(int)
        except (ValueError, AttributeError):
            # Si no funciona, intentar conversión directa
            df_trabajo['año_numerico'] = pd.to_numeric(df_trabajo['campaña'], errors='coerce')

        # Filtrar valores válidos
        df_trabajo = df_trabajo.dropna(subset=['año_numerico'])
        if len(df_trabajo) < 10:
            messagebox.showwarning("Advertencia", "No hay suficientes datos válidos después del procesamiento de las campañas.")
            return
        df_trabajo['año_numerico'] = df_trabajo['año_numerico'].astype(int)

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

        # Definir modelos y parámetros para comparación
        models = {
            'SVR RBF': {
                'model': SVR(),
                'params': {
                    'kernel': ['rbf'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                    'epsilon': [0.01, 0.1, 0.2]
                }
            },
            'SVR Lineal': {
                'model': SVR(),
                'params': {
                    'kernel': ['linear'],
                    'C': [0.1, 1, 10, 100],
                    'epsilon': [0.01, 0.1, 0.2]
                }
            },
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            }
        }

        # Entrenar y evaluar modelos
        results = {}
        best_model = None
        best_score = -float('inf')
        best_model_name = ""

        print("🔍 Optimizando modelos de IA...")

        for name, config in models.items():
            try:
                # Grid Search con validación cruzada
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )

                grid_search.fit(X_train, y_train)

                # Evaluar en conjunto de prueba
                y_pred_scaled = grid_search.predict(X_test)
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

                # Calcular métricas
                mse = mean_squared_error(y_test_orig, y_pred)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_test_orig - y_pred))
                r2 = r2_score(y_test_orig, y_pred)

                results[name] = {
                    'model': grid_search.best_estimator_,
                    'params': grid_search.best_params_,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'y_pred': y_pred,
                    'cv_score': -grid_search.best_score_
                }

                # Actualizar mejor modelo
                if r2 > best_score:
                    best_score = r2
                    best_model = grid_search.best_estimator_
                    best_model_name = name

                print(f"✅ {name}: R² = {r2:.3f}, RMSE = {rmse:.2f}")

            except Exception as e:
                print(f"❌ Error en {name}: {e}")
                continue

        if not results:
            messagebox.showerror("Error", "No se pudieron entrenar los modelos correctamente.")
            return

        # Crear visualización comparativa
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Gráfico 1: Comparación de modelos
        model_names = list(results.keys())
        r2_scores = [results[name]['r2'] for name in model_names]
        rmse_scores = [results[name]['rmse'] for name in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        bars1 = ax1.bar(x - width/2, r2_scores, width, label='R²', color='skyblue', alpha=0.8)
        ax1.set_ylabel('Coeficiente de Determinación (R²)', color='skyblue')
        ax1.set_title('Comparación de Rendimiento de Modelos')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45)
        ax1.legend(loc='upper left')

        ax1_twin = ax1.twinx()
        bars2 = ax1_twin.bar(x + width/2, rmse_scores, width, label='RMSE', color='orange', alpha=0.8)
        ax1_twin.set_ylabel('Error Cuadrático Medio (RMSE)', color='orange')
        ax1_twin.legend(loc='upper right')

        # Agregar valores en barras
        for bar, val in zip(bars1, r2_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        for bar, val in zip(bars2, rmse_scores):
            ax1_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.02,
                         f'{val:.0f}', ha='center', va='bottom', fontsize=8)

        # Gráfico 2: Predicciones vs Valores Reales (mejor modelo)
        best_result = results[best_model_name]
        ax2.scatter(y_test_orig, best_result['y_pred'], alpha=0.6, color='green', s=50)
        ax2.plot([y_test_orig.min(), y_test_orig.max()],
                [y_test_orig.min(), y_test_orig.max()],
                'r--', linewidth=2, label='Línea ideal')
        ax2.set_xlabel('Producción Real (toneladas)')
        ax2.set_ylabel('Producción Predicha (toneladas)')
        ax2.set_title(f'Predicciones vs Realidad - {best_model_name}')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Agregar línea de tendencia
        z = np.polyfit(y_test_orig, best_result['y_pred'], 1)
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
        errores = y_test_orig - best_result['y_pred']
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
        best_result = results[best_model_name]

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
            f"🤖 ANÁLISIS PREDICTIVO AVANZADO CON IA\n\n"
            f"📊 Datos analizados: {total_datos:,} registros de producción\n"
            f"📅 Período: {años_sorted[0]} - {años_sorted[-1]} ({años_unicos} campañas)\n"
            f"🌾 Producción total: {produccion_total:,.0f} toneladas\n\n"
            f"🏆 MEJOR MODELO: {best_model_name}\n"
            f"   • Parámetros óptimos: {best_result['params']}\n"
            f"   • R² (precisión): {best_result['r2']:.3f}\n"
            f"   • RMSE (error): {best_result['rmse']:.0f} toneladas\n"
            f"   • MAE (error absoluto): {best_result['mae']:.0f} toneladas\n\n"
            f"📈 COMPARACIÓN DE MODELOS:\n"
        )

        for name, result in results.items():
            icon = "🏆" if name == best_model_name else "📊"
            explanation += f"   {icon} {name}: R²={result['r2']:.3f}, RMSE={result['rmse']:.0f}\n"

        explanation += (
            f"\n🔮 PREDICCIONES FUTURAS (5 años):\n"
            f"   • {años_futuros[0]}: {y_futuro[0]:,.0f} toneladas\n"
            f"   • {años_futuros[1]}: {y_futuro[1]:,.0f} toneladas\n"
            f"   • {años_futuros[2]}: {y_futuro[2]:,.0f} toneladas\n"
            f"   • {años_futuros[3]}: {y_futuro[3]:,.0f} toneladas\n"
            f"   • {años_futuros[4]}: {y_futuro[4]:,.0f} toneladas\n\n"
            f"📊 ANÁLISIS DE TENDENCIAS:\n"
            f"   • Cambio total en el período: {cambio_total:+.1f}%\n"
            f"   • Tendencia: {'📈 Crecimiento' if cambio_total > 5 else '📉 Declive' if cambio_total < -5 else '➡️ Estable'}\n\n"
            f"🎯 INTERPRETACIÓN PARA TESIS:\n"
            f"   • El modelo {best_model_name} explica el {best_result['r2']*100:.1f}% de la variabilidad en producción\n"
            f"   • Las predicciones futuras muestran {('continuidad' if abs(cambio_total) < 10 else 'cambio significativo')}\n"
            f"   • La IA identifica patrones no lineales que métodos tradicionales no capturan\n"
            f"   • Los errores de predicción ({best_result['rmse']:.0f} ton) indican precisión aceptable\n\n"
            f"💡 APLICACIONES PRÁCTICAS:\n"
            f"   • Planificación agrícola basada en predicciones precisas\n"
            f"   • Optimización de recursos con proyecciones futuras\n"
            f"   • Gestión de riesgos con escenarios predictivos\n"
            f"   • Toma de decisiones estratégicas fundamentada en IA"
        )

        messagebox.showinfo("Predicción Avanzada de Tendencias con IA",
                          f"Análisis completado y guardado en {output_file}\n\n{explanation}")

    def analisis_predictivo_nn(self):
        """Realiza un análisis predictivo utilizando una red neuronal simple."""
        if self.df.empty or 'sup_sembrada' not in self.df.columns or 'sup_cosechada' not in self.df.columns or 'rendimiento' not in self.df.columns or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "El DataFrame debe contener las columnas 'sup_sembrada', 'sup_cosechada', 'rendimiento' y 'produccion'.")
            return

        # Preparar datos
        features = self.df[['sup_sembrada', 'sup_cosechada', 'rendimiento']]
        target = self.df['produccion']

        # Escalado de características
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
        target_scaled = scaler.fit_transform(target.values.reshape(-1, 1)).ravel()

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.2, random_state=42)

        # Construir modelo de red neuronal
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        # Compilar modelo
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Entrenar modelo
        model.fit(X_train, y_train, epochs=100, validation_split=0.2)

        # Evaluar modelo
        loss = model.evaluate(X_test, y_test)
        logging.info(f"Pérdida en el conjunto de prueba: {loss}")

        # Predicciones
        predictions = model.predict(X_test)
        predictions_rescaled = scaler.inverse_transform(predictions)  # Reescalar las predicciones al rango original

        # Mostrar algunas predicciones
        logging.info(f"Algunas predicciones reescaladas: {predictions_rescaled[:5]}")

        explanation = (
            "Este análisis utiliza una red neuronal simple para predecir la producción basada en la superficie sembrada, "
            "superficie cosechada y rendimiento. Las predicciones se reescalan al rango original para interpretación."
        )
        messagebox.showinfo("Análisis Predictivo con Red Neuronal", explanation)

    def geocodificar_direcciones(self):
        """Geocodifica direcciones con barra de progreso moderna y guarda las coordenadas en el DataFrame."""
        if self.df.empty or 'departamento' not in self.df.columns or 'provincia' not in self.df.columns or 'pais' not in self.df.columns:
            messagebox.showwarning("Advertencia", "Por favor, asegúrese de que el archivo CSV contenga las columnas 'departamento', 'provincia' y 'pais'.")
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
            "Este proceso geocodifica las direcciones de las localidades con una barra de progreso moderna, "
            "agregando coordenadas geográficas (latitud y longitud) al DataFrame. "
            "Esto es útil para análisis geoespaciales y visualización de datos en mapas."
        )
        
        messagebox.showinfo("Geocodificación", 
                           f"Geocodificación completada.\n"
                           f"Direcciones procesadas: {total_rows}\n"
                           f"Geocodificaciones exitosas: {successful_geocodes}\n"
                           f"Geocodificaciones fallidas: {failed_geocodes}\n"
                           f"Archivo guardado en: {geocoded_file}\n\n{explanation}")

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
                    popup=f"{row['GeocodedAddress']} - Cultivo: {row['cultivo']}",
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
        plt.suptitle("produccion_top_cultivos", fontsize=10, y=0.98, ha='left', x=0.02, style='italic', alpha=0.7)
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
