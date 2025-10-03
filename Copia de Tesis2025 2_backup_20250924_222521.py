import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import folium
import webbrowser
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from time import sleep
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

geolocator = Nominatim(user_agent="geopy/1.22.0 (github.com/geopy/geopy)")

class DataAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Aplicación de Análisis de Datos")
        self.df = pd.DataFrame()
        self.mapa_generado = False

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
        self.analisis_menu.add_command(label="Análisis Geoespacial", command=self.analisis_geoespacial)
        self.analisis_menu.add_command(label="Análisis de Correlación", command=self.analisis_correlacion)
        self.analisis_menu.add_command(label="Modelos Predictivos", command=self.modelos_predictivos)
        self.analisis_menu.add_command(label="Análisis de Riesgos", command=self.analisis_riesgos)
        self.analisis_menu.add_command(label="Correlación Sup. Sembrada-Sup. Cosechada", command=self.correlacion_superficie_produccion)
        self.analisis_menu.add_command(label="Producción Total por Provincia", command=self.produccion_total_por_provincia)

        self.geocodificacion_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Geocodificación", menu=self.geocodificacion_menu)
        self.geocodificacion_menu.add_command(label="Geocodificar Direcciones", command=self.geocodificar_direcciones)
        self.geocodificacion_menu.add_command(label="Generar Mapa", command=self.generar_mapa)

        # Crear frame principal para los botones
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título
        title_label = tk.Label(self.main_frame, text="Aplicación de Análisis de Datos", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))

        # Frame para botones de archivo
        file_frame = tk.LabelFrame(self.main_frame, text="Archivo", font=("Arial", 12, "bold"))
        file_frame.pack(fill=tk.X, pady=(0, 10))

        cargar_btn = tk.Button(file_frame, text="Cargar CSV", command=self.cargar_csv,
                              bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                              width=15, height=2)
        cargar_btn.pack(side=tk.LEFT, padx=10, pady=10)

        # Frame para botones de análisis
        analysis_frame = tk.LabelFrame(self.main_frame, text="Análisis", font=("Arial", 12, "bold"))
        analysis_frame.pack(fill=tk.X, pady=(0, 10))

        # Primera fila de botones de análisis
        analysis_row1 = tk.Frame(analysis_frame)
        analysis_row1.pack(fill=tk.X, pady=5)

        sumar_btn = tk.Button(analysis_row1, text="Sumar Columnas", command=self.sumar_columnas,
                             bg="#2196F3", fg="white", font=("Arial", 9), width=18, height=2)
        sumar_btn.pack(side=tk.LEFT, padx=5, pady=5)

        temporal_btn = tk.Button(analysis_row1, text="Análisis Temporal", command=self.analisis_temporal,
                                bg="#2196F3", fg="white", font=("Arial", 9), width=18, height=2)
        temporal_btn.pack(side=tk.LEFT, padx=5, pady=5)

        correlacion_btn = tk.Button(analysis_row1, text="Análisis de Correlación", command=self.analisis_correlacion,
                                   bg="#2196F3", fg="white", font=("Arial", 9), width=18, height=2)
        correlacion_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Segunda fila de botones de análisis
        analysis_row2 = tk.Frame(analysis_frame)
        analysis_row2.pack(fill=tk.X, pady=5)

        produccion_btn = tk.Button(analysis_row2, text="Producción por Provincia", command=self.produccion_total_por_provincia,
                                  bg="#2196F3", fg="white", font=("Arial", 9), width=18, height=2)
        produccion_btn.pack(side=tk.LEFT, padx=5, pady=5)

        superficie_btn = tk.Button(analysis_row2, text="Correlación Superficie", command=self.correlacion_superficie_produccion,
                                  bg="#2196F3", fg="white", font=("Arial", 9), width=18, height=2)
        superficie_btn.pack(side=tk.LEFT, padx=5, pady=5)

        riesgos_btn = tk.Button(analysis_row2, text="Análisis de Riesgos", command=self.analisis_riesgos,
                               bg="#2196F3", fg="white", font=("Arial", 9), width=18, height=2)
        riesgos_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Frame para botones de geocodificación
        geo_frame = tk.LabelFrame(self.main_frame, text="Geocodificación", font=("Arial", 12, "bold"))
        geo_frame.pack(fill=tk.X, pady=(0, 10))

        geocode_btn = tk.Button(geo_frame, text="Geocodificar Direcciones", command=self.geocodificar_direcciones,
                               bg="#FF9800", fg="white", font=("Arial", 10, "bold"), width=20, height=2)
        geocode_btn.pack(side=tk.LEFT, padx=10, pady=10)

        mapa_btn = tk.Button(geo_frame, text="Generar Mapa", command=self.generar_mapa,
                            bg="#FF9800", fg="white", font=("Arial", 10, "bold"), width=20, height=2)
        mapa_btn.pack(side=tk.LEFT, padx=10, pady=10)

        geoespacial_btn = tk.Button(geo_frame, text="Análisis Geoespacial", command=self.analisis_geoespacial,
                                   bg="#FF9800", fg="white", font=("Arial", 10, "bold"), width=20, height=2)
        geoespacial_btn.pack(side=tk.LEFT, padx=10, pady=10)

        # Frame para botones adicionales
        extra_frame = tk.LabelFrame(self.main_frame, text="Modelos y Predicción", font=("Arial", 12, "bold"))
        extra_frame.pack(fill=tk.X, pady=(0, 10))

        modelos_btn = tk.Button(extra_frame, text="Modelos Predictivos", command=self.modelos_predictivos,
                               bg="#9C27B0", fg="white", font=("Arial", 10, "bold"), width=20, height=2)
        modelos_btn.pack(side=tk.LEFT, padx=10, pady=10)

        # Botón adicional para análisis rápido
        analisis_rapido_btn = tk.Button(extra_frame, text="Análisis Rápido", command=self.analisis_rapido,
                                       bg="#FF5722", fg="white", font=("Arial", 10, "bold"), width=20, height=2)
        analisis_rapido_btn.pack(side=tk.LEFT, padx=10, pady=10)

        # Configurar tamaño mínimo de ventana
        self.root.minsize(800, 600)

    def cargar_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            logging.debug(f"Archivo CSV cargado: {file_path}")
            messagebox.showinfo("Cargar CSV", "Archivo CSV cargado exitosamente.")

    def produccion_total_por_provincia(self):
        if self.df.empty or 'provincia' not in self.df.columns or 'produccion' not in self.df.columns:
            messagebox.showwarning("Advertencia", "Por favor, asegúrese de que el archivo CSV contenga las columnas 'provincia' y 'produccion'.")
            return

        produccion_por_provincia = self.df.groupby('provincia')['produccion'].sum().sort_values(ascending=False)

        # Crear gráfico de torta
        plt.figure(figsize=(10, 8))
        plt.pie(produccion_por_provincia.values, labels=produccion_por_provincia.index, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')  # Para que el gráfico sea circular
        plt.title("Producción Total por Provincia")
        plt.tight_layout()

        produccion_por_provincia_file = OUTPUT_DIR / "produccion_por_provincia.png"
        plt.savefig(produccion_por_provincia_file)
        plt.show()
        logging.debug(f"Gráfica de producción total por provincia guardada en {produccion_por_provincia_file}")

        messagebox.showinfo("Producción Total por Provincia", f"Gráfica de producción total por provincia guardada en {produccion_por_provincia_file}")

    def correlacion_superficie_produccion(self):
        if self.df.empty or 'provincia' not in self.df.columns:
            messagebox.showwarning("Advertencia", "Por favor, asegúrese de que el archivo CSV contenga la columna 'provincia'.")
            return

        # Verificar que existan las columnas necesarias
        columnas_requeridas = ['sup_sembrada', 'sup_cosechada']
        columnas_faltantes = [col for col in columnas_requeridas if col not in self.df.columns]
        
        if columnas_faltantes:
            messagebox.showwarning("Advertencia", 
                                 f"Las siguientes columnas no están en el archivo CSV: {', '.join(columnas_faltantes)}")
            return

        try:
            # Crear ventana de selección mejorada
            self.crear_ventana_correlacion_superficie()
            
        except Exception as e:
            logging.error(f"Error en correlación de superficie: {e}")
            messagebox.showerror("Error", f"Error al ejecutar correlación de superficie: {str(e)}")

    def crear_ventana_correlacion_superficie(self):
        """Crear ventana para configurar análisis de correlación de superficie"""
        corr_window = tk.Toplevel(self.root)
        corr_window.title("Análisis de Correlación de Superficie")
        corr_window.geometry("500x400")
        corr_window.resizable(True, True)

        # Frame principal
        main_frame = tk.Frame(corr_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título
        title_label = tk.Label(main_frame, text="Correlación Superficie Sembrada vs Cosechada", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))

        # Selección de alcance del análisis
        scope_frame = tk.LabelFrame(main_frame, text="Alcance del Análisis", font=("Arial", 10, "bold"))
        scope_frame.pack(fill=tk.X, pady=(0, 10))

        self.analysis_scope = tk.StringVar(value="todas")
        
        todas_rb = tk.Radiobutton(scope_frame, text="Todas las Provincias", 
                                 variable=self.analysis_scope, value="todas")
        todas_rb.pack(anchor="w", padx=10, pady=5)
        
        provincia_rb = tk.Radiobutton(scope_frame, text="Provincia Específica", 
                                     variable=self.analysis_scope, value="provincia")
        provincia_rb.pack(anchor="w", padx=10, pady=2)

        # Selección de provincia (si es necesario)
        provincia_frame = tk.LabelFrame(main_frame, text="Seleccionar Provincia", font=("Arial", 10, "bold"))
        provincia_frame.pack(fill=tk.X, pady=(0, 10))

        provincias = self.df['provincia'].unique()
        self.selected_provincia = tk.StringVar()
        provincia_combo = ttk.Combobox(provincia_frame, textvariable=self.selected_provincia, 
                                      values=list(provincias), state="readonly")
        provincia_combo.pack(pady=10, padx=10, fill=tk.X)
        if len(provincias) > 0:
            provincia_combo.current(0)

        # Opciones de análisis
        options_frame = tk.LabelFrame(main_frame, text="Opciones de Análisis", font=("Arial", 10, "bold"))
        options_frame.pack(fill=tk.X, pady=(0, 10))

        self.include_regression = tk.BooleanVar(value=True)
        regression_cb = tk.Checkbutton(options_frame, text="Incluir línea de regresión", 
                                      variable=self.include_regression)
        regression_cb.pack(anchor="w", padx=10, pady=5)

        self.show_statistics = tk.BooleanVar(value=True)
        stats_cb = tk.Checkbutton(options_frame, text="Mostrar estadísticas detalladas", 
                                 variable=self.show_statistics)
        stats_cb.pack(anchor="w", padx=10, pady=2)

        self.temporal_analysis = tk.BooleanVar(value=False)
        temporal_cb = tk.Checkbutton(options_frame, text="Análisis temporal (por campaña)", 
                                    variable=self.temporal_analysis)
        temporal_cb.pack(anchor="w", padx=10, pady=2)

        # Botones
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))

        execute_btn = tk.Button(button_frame, text="Ejecutar Análisis", 
                               command=lambda: self.ejecutar_correlacion_superficie(corr_window),
                               bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        execute_btn.pack(side=tk.LEFT, padx=(0, 10))

        cancel_btn = tk.Button(button_frame, text="Cancelar", 
                              command=corr_window.destroy,
                              bg="#f44336", fg="white", font=("Arial", 10, "bold"))
        cancel_btn.pack(side=tk.LEFT)

    def ejecutar_correlacion_superficie(self, window):
        """Ejecutar el análisis de correlación de superficie"""
        try:
            scope = self.analysis_scope.get()
            include_regression = self.include_regression.get()
            show_statistics = self.show_statistics.get()
            temporal_analysis = self.temporal_analysis.get()
            
            # Filtrar datos según el alcance
            if scope == "provincia":
                provincia_seleccionada = self.selected_provincia.get()
                if not provincia_seleccionada:
                    messagebox.showwarning("Advertencia", "Por favor seleccione una provincia.")
                    return
                df_analisis = self.df[self.df['provincia'] == provincia_seleccionada].copy()
                titulo_base = f"Correlación de Superficie - {provincia_seleccionada}"
            else:
                df_analisis = self.df.copy()
                titulo_base = "Correlación de Superficie - Todas las Provincias"
            
            # Limpiar datos
            df_clean = df_analisis[['sup_sembrada', 'sup_cosechada']].dropna()
            
            if df_clean.empty:
                messagebox.showwarning("Advertencia", "No hay datos válidos para el análisis.")
                return
            
            # Realizar análisis
            if temporal_analysis and 'campaña' in df_analisis.columns:
                self.analisis_correlacion_temporal(df_analisis, titulo_base, include_regression, show_statistics)
            else:
                self.analisis_correlacion_simple(df_clean, titulo_base, include_regression, show_statistics)
            
            window.destroy()
            
        except Exception as e:
            logging.error(f"Error ejecutando correlación: {e}")
            messagebox.showerror("Error", f"Error al ejecutar correlación: {str(e)}")

    def analisis_correlacion_simple(self, df_clean, titulo, include_regression, show_statistics):
        """Realizar análisis de correlación simple"""
        
        # Calcular estadísticas
        correlacion = df_clean['sup_sembrada'].corr(df_clean['sup_cosechada'])
        
        # Crear figura
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfico de dispersión
        ax1 = axes[0]
        scatter = ax1.scatter(df_clean['sup_sembrada'], df_clean['sup_cosechada'], 
                             alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5)
        
        # Línea de regresión si está habilitada
        if include_regression:
            slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean['sup_sembrada'], 
                                                                          df_clean['sup_cosechada'])
            line = slope * df_clean['sup_sembrada'] + intercept
            ax1.plot(df_clean['sup_sembrada'], line, 'r-', linewidth=2, 
                    label=f'Regresión (R² = {r_value**2:.3f})')
            ax1.legend()
        
        ax1.set_xlabel('Superficie Sembrada (ha)')
        ax1.set_ylabel('Superficie Cosechada (ha)')
        ax1.set_title(f'{titulo}\nCorrelación: {correlacion:.3f}')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de barras comparativo por provincia (si aplica)
        ax2 = axes[1]
        if 'provincia' in self.df.columns and len(self.df['provincia'].unique()) > 1:
            # Agrupar por provincia
            df_grouped = self.df.groupby('provincia')[['sup_sembrada', 'sup_cosechada']].sum()
            
            x = np.arange(len(df_grouped))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, df_grouped['sup_sembrada'], width, 
                           label='Superficie Sembrada', alpha=0.8, color='lightblue')
            bars2 = ax2.bar(x + width/2, df_grouped['sup_cosechada'], width, 
                           label='Superficie Cosechada', alpha=0.8, color='lightcoral')
            
            ax2.set_xlabel('Provincia')
            ax2.set_ylabel('Superficie (ha)')
            ax2.set_title('Comparación por Provincia')
            ax2.set_xticks(x)
            ax2.set_xticklabels(df_grouped.index, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Añadir valores en las barras
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.0f}', ha='center', va='bottom', fontsize=8)
        else:
            # Histograma de diferencias
            diferencias = df_clean['sup_sembrada'] - df_clean['sup_cosechada']
            ax2.hist(diferencias, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax2.set_xlabel('Diferencia (Sembrada - Cosechada)')
            ax2.set_ylabel('Frecuencia')
            ax2.set_title('Distribución de Diferencias')
            ax2.grid(True, alpha=0.3)
            ax2.axvline(diferencias.mean(), color='red', linestyle='--', 
                       label=f'Media: {diferencias.mean():.1f}')
            ax2.legend()
        
        plt.tight_layout()
        
        # Guardar gráfico
        correlacion_file = OUTPUT_DIR / "correlacion_superficie.png"
        plt.savefig(correlacion_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Mostrar estadísticas si está habilitado
        if show_statistics:
            self.mostrar_estadisticas_correlacion(df_clean, correlacion, include_regression)
        
        logging.debug(f"Análisis de correlación guardado en {correlacion_file}")
        messagebox.showinfo("Correlación de Superficie", 
                           f"Análisis completado exitosamente.\n"
                           f"Correlación: {correlacion:.3f}\n"
                           f"Gráfico guardado en: {correlacion_file}")

    def analisis_correlacion_temporal(self, df_analisis, titulo, include_regression, show_statistics):
        """Realizar análisis de correlación temporal por campaña"""
        
        # Extraer año de campaña
        df_temp = df_analisis.copy()
        df_temp['año'] = df_temp['campaña'].str.split('/').str[0].astype(int)
        
        # Agrupar por año
        df_temporal = df_temp.groupby('año')[['sup_sembrada', 'sup_cosechada']].sum()
        
        # Crear figura
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Gráfico temporal
        ax1 = axes[0, 0]
        ax1.plot(df_temporal.index, df_temporal['sup_sembrada'], 'b-o', 
                label='Superficie Sembrada', linewidth=2, markersize=4)
        ax1.plot(df_temporal.index, df_temporal['sup_cosechada'], 'r-s', 
                label='Superficie Cosechada', linewidth=2, markersize=4)
        ax1.set_xlabel('Campaña')
        ax1.set_ylabel('Superficie (ha)')
        ax1.set_title(f'{titulo} - Evolución Temporal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Correlación temporal
        ax2 = axes[0, 1]
        correlacion_temporal = df_temporal['sup_sembrada'].corr(df_temporal['sup_cosechada'])
        scatter = ax2.scatter(df_temporal['sup_sembrada'], df_temporal['sup_cosechada'], 
                             alpha=0.7, s=60, c=df_temporal.index, cmap='viridis', 
                             edgecolors='black', linewidth=0.5)
        
        if include_regression:
            slope, intercept, r_value, p_value, std_err = stats.linregress(df_temporal['sup_sembrada'], 
                                                                          df_temporal['sup_cosechada'])
            line = slope * df_temporal['sup_sembrada'] + intercept
            ax2.plot(df_temporal['sup_sembrada'], line, 'r-', linewidth=2, 
                    label=f'Regresión (R² = {r_value**2:.3f})')
            ax2.legend()
        
        ax2.set_xlabel('Superficie Sembrada (ha)')
        ax2.set_ylabel('Superficie Cosechada (ha)')
        ax2.set_title(f'Correlación Temporal: {correlacion_temporal:.3f}')
        ax2.grid(True, alpha=0.3)
        
        # Colorbar para los años
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Campaña')
        
        # Eficiencia de cosecha (cosechada/sembrada)
        ax3 = axes[1, 0]
        eficiencia = (df_temporal['sup_cosechada'] / df_temporal['sup_sembrada']) * 100
        ax3.plot(df_temporal.index, eficiencia, 'g-o', linewidth=2, markersize=4)
        ax3.set_xlabel('Campaña')
        ax3.set_ylabel('Eficiencia de Cosecha (%)')
        ax3.set_title('Eficiencia de Cosecha por Campaña')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(eficiencia.mean(), color='red', linestyle='--', 
                   label=f'Promedio: {eficiencia.mean():.1f}%')
        ax3.legend()
        
        # Diferencias absolutas
        ax4 = axes[1, 1]
        diferencias = df_temporal['sup_sembrada'] - df_temporal['sup_cosechada']
        ax4.bar(df_temporal.index, diferencias, alpha=0.7, 
               color=['red' if x > 0 else 'blue' for x in diferencias])
        ax4.set_xlabel('Campaña')
        ax4.set_ylabel('Diferencia (Sembrada - Cosechada)')
        ax4.set_title('Diferencias por Campaña')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(0, color='black', linestyle='-', linewidth=1)
        
        plt.tight_layout()
        
        # Guardar gráfico
        correlacion_temporal_file = OUTPUT_DIR / "correlacion_superficie_temporal.png"
        plt.savefig(correlacion_temporal_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Mostrar estadísticas si está habilitado
        if show_statistics:
            self.mostrar_estadisticas_correlacion_temporal(df_temporal, correlacion_temporal, eficiencia)
        
        logging.debug(f"Análisis de correlación temporal guardado en {correlacion_temporal_file}")
        messagebox.showinfo("Correlación de Superficie Temporal", 
                           f"Análisis temporal completado exitosamente.\n"
                           f"Correlación: {correlacion_temporal:.3f}\n"
                           f"Eficiencia promedio: {eficiencia.mean():.1f}%\n"
                           f"Gráfico guardado en: {correlacion_temporal_file}")

    def mostrar_estadisticas_correlacion(self, df_clean, correlacion, include_regression):
        """Mostrar ventana con estadísticas detalladas de correlación"""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Estadísticas de Correlación")
        stats_window.geometry("500x400")
        
        main_frame = tk.Frame(stats_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        title_label = tk.Label(main_frame, text="Estadísticas Detalladas", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Estadísticas básicas
        stats_text = f"ESTADÍSTICAS DE CORRELACIÓN\n\n"
        stats_text += f"Coeficiente de Correlación de Pearson: {correlacion:.4f}\n"
        
        # Interpretación de la correlación
        if abs(correlacion) >= 0.8:
            interpretacion = "Muy fuerte"
        elif abs(correlacion) >= 0.6:
            interpretacion = "Fuerte"
        elif abs(correlacion) >= 0.4:
            interpretacion = "Moderada"
        elif abs(correlacion) >= 0.2:
            interpretacion = "Débil"
        else:
            interpretacion = "Muy débil"
        
        stats_text += f"Interpretación: Correlación {interpretacion}\n\n"
        
        # Estadísticas descriptivas
        stats_text += f"SUPERFICIE SEMBRADA:\n"
        stats_text += f"  Media: {df_clean['sup_sembrada'].mean():.2f} ha\n"
        stats_text += f"  Mediana: {df_clean['sup_sembrada'].median():.2f} ha\n"
        stats_text += f"  Desv. Estándar: {df_clean['sup_sembrada'].std():.2f} ha\n"
        stats_text += f"  Mínimo: {df_clean['sup_sembrada'].min():.2f} ha\n"
        stats_text += f"  Máximo: {df_clean['sup_sembrada'].max():.2f} ha\n\n"
        
        stats_text += f"SUPERFICIE COSECHADA:\n"
        stats_text += f"  Media: {df_clean['sup_cosechada'].mean():.2f} ha\n"
        stats_text += f"  Mediana: {df_clean['sup_cosechada'].median():.2f} ha\n"
        stats_text += f"  Desv. Estándar: {df_clean['sup_cosechada'].std():.2f} ha\n"
        stats_text += f"  Mínimo: {df_clean['sup_cosechada'].min():.2f} ha\n"
        stats_text += f"  Máximo: {df_clean['sup_cosechada'].max():.2f} ha\n\n"
        
        # Estadísticas de regresión si está habilitada
        if include_regression:
            slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean['sup_sembrada'], 
                                                                          df_clean['sup_cosechada'])
            stats_text += f"REGRESIÓN LINEAL:\n"
            stats_text += f"  Pendiente: {slope:.4f}\n"
            stats_text += f"  Intercepto: {intercept:.2f}\n"
            stats_text += f"  R²: {r_value**2:.4f}\n"
            stats_text += f"  P-valor: {p_value:.4e}\n"
            stats_text += f"  Error estándar: {std_err:.4f}\n"
        
        stats_label = tk.Label(main_frame, text=stats_text, font=("Arial", 9), 
                              justify=tk.LEFT, bg="white")
        stats_label.pack(anchor="w", fill=tk.BOTH, expand=True)
        
        close_btn = tk.Button(main_frame, text="Cerrar", command=stats_window.destroy,
                             bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        close_btn.pack(pady=(20, 0))

    def mostrar_estadisticas_correlacion_temporal(self, df_temporal, correlacion, eficiencia):
        """Mostrar estadísticas de correlación temporal"""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Estadísticas de Correlación Temporal")
        stats_window.geometry("500x450")
        
        main_frame = tk.Frame(stats_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        title_label = tk.Label(main_frame, text="Estadísticas Temporales Detalladas", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        stats_text = f"ANÁLISIS TEMPORAL DE CORRELACIÓN\n\n"
        stats_text += f"Período analizado: {df_temporal.index.min()} - {df_temporal.index.max()}\n"
        stats_text += f"Número de campañas: {len(df_temporal)}\n"
        stats_text += f"Correlación temporal: {correlacion:.4f}\n\n"
        
        stats_text += f"EFICIENCIA DE COSECHA:\n"
        stats_text += f"  Promedio: {eficiencia.mean():.2f}%\n"
        stats_text += f"  Mediana: {eficiencia.median():.2f}%\n"
        stats_text += f"  Desv. Estándar: {eficiencia.std():.2f}%\n"
        stats_text += f"  Mínimo: {eficiencia.min():.2f}% (Campaña {eficiencia.idxmin()})\n"
        stats_text += f"  Máximo: {eficiencia.max():.2f}% (Campaña {eficiencia.idxmax()})\n\n"
        
        # Tendencias
        diferencias = df_temporal['sup_sembrada'] - df_temporal['sup_cosechada']
        stats_text += f"DIFERENCIAS (Sembrada - Cosechada):\n"
        stats_text += f"  Promedio: {diferencias.mean():.2f} ha\n"
        stats_text += f"  Total acumulado: {diferencias.sum():.2f} ha\n"
        stats_text += f"  Campañas con pérdidas: {(diferencias > 0).sum()}\n"
        stats_text += f"  Campañas con ganancias: {(diferencias < 0).sum()}\n"
        
        stats_label = tk.Label(main_frame, text=stats_text, font=("Arial", 9), 
                              justify=tk.LEFT, bg="white")
        stats_label.pack(anchor="w", fill=tk.BOTH, expand=True)
        
        close_btn = tk.Button(main_frame, text="Cerrar", command=stats_window.destroy,
                             bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        close_btn.pack(pady=(20, 0))

    def ask_option(self, title, message, options):
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("250x150")
        dialog.resizable(False, False)

        label = tk.Label(dialog, text=message)
        label.pack(pady=10)

        combobox_value = tk.StringVar()
        combobox = ttk.Combobox(dialog, textvariable=combobox_value, values=options)
        combobox.pack(pady=10)
        combobox.current(0)  # Seleccionar el primer elemento por defecto

        button = tk.Button(dialog, text="Aceptar", command=dialog.destroy)
        button.pack(pady=10)

        dialog.grab_set()
        dialog.wait_window()

        return combobox_value.get()

    def sumar_columnas(self):
        if self.df.empty:
            messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return

        suma_columnas = self.df.select_dtypes(include=[float, int]).sum()
        suma_columnas.plot(kind='bar')
        plt.title("Suma de Columnas Numéricas")
        plt.ylabel("Suma")
        plt.xlabel("Columnas")
        plt.tight_layout()

        suma_columnas_file = OUTPUT_DIR / "suma_columnas.png"
        plt.savefig(suma_columnas_file)
        plt.show()
        logging.debug(f"Gráfica de suma de columnas guardada en {suma_columnas_file}")

        messagebox.showinfo("Suma de Columnas", f"Gráfica de suma de columnas guardada en {suma_columnas_file}")

    def analisis_temporal(self):
        if self.df.empty or 'campaña' not in self.df.columns:
            messagebox.showwarning("Advertencia", "Por favor, asegúrese de que el archivo CSV contenga una columna 'campaña'.")
            return

        try:
            # Crear una copia del DataFrame para no modificar el original
            df_temp = self.df.copy()
            
            # Extraer el primer año de la columna 'campaña' (formato '1969/1970')
            df_temp['año'] = df_temp['campaña'].str.split('/').str[0].astype(int)
            
            # Obtener solo las columnas numéricas (excluyendo la nueva columna 'año')
            numeric_columns = df_temp.select_dtypes(include=[np.number]).columns.tolist()
            if 'año' in numeric_columns:
                numeric_columns.remove('año')
            
            if not numeric_columns:
                messagebox.showwarning("Advertencia", "No se encontraron columnas numéricas para el análisis temporal.")
                return
            
            # Filtrar solo las columnas relevantes para agricultura si existen
            columnas_agricolas = ['sup_sembrada', 'sup_cosechada', 'produccion', 'rendimiento']
            columnas_disponibles = [col for col in columnas_agricolas if col in numeric_columns]
            
            if columnas_disponibles:
                columnas_a_usar = columnas_disponibles
            else:
                # Si no hay columnas agrícolas específicas, usar todas las columnas numéricas disponibles
                columnas_a_usar = numeric_columns
            
            # Agrupar por año y sumar solo las columnas seleccionadas
            df_temporal = df_temp.groupby('año')[columnas_a_usar].sum()
            
            # Verificar que hay datos para graficar
            if df_temporal.empty or df_temporal.sum().sum() == 0:
                messagebox.showwarning("Advertencia", "No hay datos válidos para el análisis temporal.")
                return
            
            # Asegurar que el índice esté ordenado cronológicamente
            df_temporal = df_temporal.sort_index()
            
            # Rellenar años faltantes con interpolación si hay gaps grandes
            años_completos = range(df_temporal.index.min(), df_temporal.index.max() + 1)
            df_temporal = df_temporal.reindex(años_completos)
            
            # Interpolar valores faltantes solo si hay pocos gaps
            missing_years = df_temporal.isnull().any(axis=1).sum()
            total_years = len(df_temporal)
            
            if missing_years / total_years < 0.3:  # Solo interpolar si menos del 30% son faltantes
                df_temporal = df_temporal.interpolate(method='linear')
            else:
                # Si hay muchos faltantes, eliminar filas completamente vacías
                df_temporal = df_temporal.dropna(how='all')
            
            # Crear el gráfico mejorado
            plt.figure(figsize=(14, 10))
            
            # Crear subplots para mejor visualización
            if len(columnas_a_usar) > 4:
                # Si hay muchas columnas, crear múltiples subplots
                n_cols = min(2, len(columnas_a_usar))
                n_rows = (len(columnas_a_usar) + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
                if n_rows == 1:
                    axes = [axes] if n_cols == 1 else axes
                else:
                    axes = axes.flatten()
                
                for i, col in enumerate(columnas_a_usar):
                    if i < len(axes):
                        ax = axes[i]
                        df_temporal[col].plot(kind='line', ax=ax, marker='o', linewidth=2, markersize=4)
                        ax.set_title(f'{col.title()}', fontsize=12, fontweight='bold')
                        ax.set_xlabel('Campaña', fontsize=10)
                        ax.set_ylabel('Valores', fontsize=10)
                        ax.grid(True, alpha=0.3)
                        
                        # Rotar etiquetas del eje x si hay muchos años
                        if len(df_temporal) > 10:
                            ax.tick_params(axis='x', rotation=45)
                
                # Ocultar subplots vacíos
                for i in range(len(columnas_a_usar), len(axes)):
                    axes[i].set_visible(False)
                
                plt.suptitle("Análisis Temporal por Variable", fontsize=16, fontweight='bold')
                plt.tight_layout()
                
            else:
                # Si hay pocas columnas, usar un solo gráfico
                ax = plt.gca()
                
                # Graficar cada serie con diferentes estilos
                colors = plt.cm.Set1(np.linspace(0, 1, len(columnas_a_usar)))
                markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
                
                for i, col in enumerate(columnas_a_usar):
                    df_temporal[col].plot(kind='line', ax=ax, 
                                        marker=markers[i % len(markers)], 
                                        linewidth=2.5, markersize=6,
                                        color=colors[i], label=col.title())
                
                plt.title("Análisis Temporal de Variables Agrícolas", fontsize=16, fontweight='bold')
                plt.ylabel("Valores", fontsize=12)
                plt.xlabel("Campaña", fontsize=12)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                
                # Mejorar formato del eje x
                if len(df_temporal) > 15:
                    plt.xticks(rotation=45)
                
                # Añadir líneas de tendencia si hay suficientes datos
                if len(df_temporal) > 5:
                    for i, col in enumerate(columnas_a_usar):
                        if not df_temporal[col].isnull().all():
                            # Calcular tendencia lineal
                            x_vals = np.arange(len(df_temporal))
                            y_vals = df_temporal[col].values
                            
                            # Eliminar NaN para el cálculo de tendencia
                            mask = ~np.isnan(y_vals)
                            if mask.sum() > 2:  # Necesitamos al menos 3 puntos
                                z = np.polyfit(x_vals[mask], y_vals[mask], 1)
                                p = np.poly1d(z)
                                plt.plot(df_temporal.index, p(x_vals), "--", 
                                        color=colors[i], alpha=0.7, linewidth=1.5)
                
                plt.tight_layout()

            analisis_temporal_file = OUTPUT_DIR / "analisis_temporal.png"
            plt.savefig(analisis_temporal_file, dpi=300, bbox_inches='tight')
            plt.show()
            logging.debug(f"Gráfica de análisis temporal guardada en {analisis_temporal_file}")

            # Mostrar información adicional mejorada
            info_msg = f"Análisis temporal completado exitosamente.\n\n"
            info_msg += f"Columnas analizadas: {', '.join(columnas_a_usar)}\n"
            info_msg += f"Período: {df_temporal.index.min()} - {df_temporal.index.max()}\n"
            info_msg += f"Total de años: {len(df_temporal)}\n"
            info_msg += f"Años con datos: {df_temporal.dropna(how='all').shape[0]}\n"
            
            # Añadir estadísticas de tendencia
            tendencias = []
            for col in columnas_a_usar:
                if not df_temporal[col].isnull().all():
                    # Calcular correlación con el tiempo para determinar tendencia
                    x_vals = np.arange(len(df_temporal))
                    y_vals = df_temporal[col].values
                    mask = ~np.isnan(y_vals)
                    if mask.sum() > 2:
                        correlation = np.corrcoef(x_vals[mask], y_vals[mask])[0, 1]
                        if correlation > 0.3:
                            tendencias.append(f"{col}: Creciente")
                        elif correlation < -0.3:
                            tendencias.append(f"{col}: Decreciente")
                        else:
                            tendencias.append(f"{col}: Estable")
            
            if tendencias:
                info_msg += f"\nTendencias detectadas:\n" + "\n".join(f"• {t}" for t in tendencias)
            
            info_msg += f"\n\nGráfica guardada en: {analisis_temporal_file}"

            messagebox.showinfo("Análisis Temporal", info_msg)
        
        except Exception as e:
            logging.error(f"Error al realizar el análisis temporal: {e}")
            messagebox.showerror("Error", f"Ocurrió un error al realizar el análisis temporal: {e}")

    def analisis_geoespacial(self):
        """Análisis geoespacial avanzado con estadísticas, clustering y patrones espaciales"""
        if self.df.empty:
            messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return

        try:
            # Crear ventana de configuración del análisis geoespacial
            self.crear_ventana_analisis_geoespacial()
            
        except Exception as e:
            logging.error(f"Error en análisis geoespacial: {e}")
            messagebox.showerror("Error", f"Error al ejecutar análisis geoespacial: {str(e)}")

    def crear_ventana_analisis_geoespacial(self):
        """Crear ventana para configurar análisis geoespacial avanzado"""
        geo_window = tk.Toplevel(self.root)
        geo_window.title("Análisis Geoespacial Avanzado")
        geo_window.geometry("600x500")
        geo_window.resizable(True, True)

        # Frame principal
        main_frame = tk.Frame(geo_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título
        title_label = tk.Label(main_frame, text="Análisis Geoespacial Avanzado", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))

        # Descripción
        desc_text = ("Este análisis va más allá de un simple mapa. Incluye:\n"
                    "• Clustering espacial y detección de patrones\n"
                    "• Análisis de densidad y distribución geográfica\n"
                    "• Estadísticas espaciales y correlaciones geográficas\n"
                    "• Identificación de zonas de alta/baja productividad")
        desc_label = tk.Label(main_frame, text=desc_text, font=("Arial", 10), 
                             justify=tk.LEFT, fg="gray")
        desc_label.pack(pady=(0, 20))

        # Verificar disponibilidad de coordenadas
        has_coordinates = ('Latitude' in self.df.columns and 'Longitude' in self.df.columns)
        
        # Selección de tipo de análisis
        analysis_frame = tk.LabelFrame(main_frame, text="Tipo de Análisis", font=("Arial", 10, "bold"))
        analysis_frame.pack(fill=tk.X, pady=(0, 10))

        self.geo_analysis_type = tk.StringVar(value="completo")
        
        if has_coordinates:
            completo_rb = tk.Radiobutton(analysis_frame, text="Análisis Completo con Coordenadas", 
                                        variable=self.geo_analysis_type, value="completo")
            completo_rb.pack(anchor="w", padx=10, pady=5)
            
            clustering_rb = tk.Radiobutton(analysis_frame, text="Solo Clustering Espacial", 
                                          variable=self.geo_analysis_type, value="clustering")
            clustering_rb.pack(anchor="w", padx=10, pady=2)
            
            densidad_rb = tk.Radiobutton(analysis_frame, text="Solo Análisis de Densidad", 
                                        variable=self.geo_analysis_type, value="densidad")
            densidad_rb.pack(anchor="w", padx=10, pady=2)
        
        # Análisis por provincias (siempre disponible)
        provincias_rb = tk.Radiobutton(analysis_frame, text="Análisis por Provincias (Sin coordenadas)", 
                                      variable=self.geo_analysis_type, value="provincias")
        provincias_rb.pack(anchor="w", padx=10, pady=2)

        # Configuración de parámetros
        params_frame = tk.LabelFrame(main_frame, text="Parámetros de Análisis", font=("Arial", 10, "bold"))
        params_frame.pack(fill=tk.X, pady=(0, 10))

        # Número de clusters
        cluster_frame = tk.Frame(params_frame)
        cluster_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(cluster_frame, text="Número de clusters:").pack(side=tk.LEFT)
        self.num_clusters = tk.IntVar(value=5)
        cluster_spin = tk.Spinbox(cluster_frame, from_=2, to=10, textvariable=self.num_clusters, width=5)
        cluster_spin.pack(side=tk.RIGHT)

        # Variable para análisis
        if len(self.df.select_dtypes(include=[np.number]).columns) > 0:
            var_frame = tk.Frame(params_frame)
            var_frame.pack(fill=tk.X, padx=10, pady=5)
            
            tk.Label(var_frame, text="Variable para análisis:").pack(side=tk.LEFT)
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.analysis_variable = tk.StringVar()
            var_combo = ttk.Combobox(var_frame, textvariable=self.analysis_variable, 
                                    values=numeric_cols, state="readonly", width=15)
            var_combo.pack(side=tk.RIGHT)
            if 'produccion' in numeric_cols:
                var_combo.set('produccion')
            elif len(numeric_cols) > 0:
                var_combo.set(numeric_cols[0])

        # Opciones adicionales
        options_frame = tk.LabelFrame(main_frame, text="Opciones Adicionales", font=("Arial", 10, "bold"))
        options_frame.pack(fill=tk.X, pady=(0, 10))

        self.include_statistics = tk.BooleanVar(value=True)
        stats_cb = tk.Checkbutton(options_frame, text="Incluir estadísticas espaciales detalladas", 
                                 variable=self.include_statistics)
        stats_cb.pack(anchor="w", padx=10, pady=5)

        self.generate_heatmap = tk.BooleanVar(value=True)
        heatmap_cb = tk.Checkbutton(options_frame, text="Generar mapas de calor", 
                                   variable=self.generate_heatmap)
        heatmap_cb.pack(anchor="w", padx=10, pady=2)

        self.export_results = tk.BooleanVar(value=True)
        export_cb = tk.Checkbutton(options_frame, text="Exportar resultados a CSV", 
                                  variable=self.export_results)
        export_cb.pack(anchor="w", padx=10, pady=2)

        # Botones
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))

        execute_btn = tk.Button(button_frame, text="Ejecutar Análisis Geoespacial", 
                               command=lambda: self.ejecutar_analisis_geoespacial_avanzado(geo_window),
                               bg="#FF9800", fg="white", font=("Arial", 11, "bold"))
        execute_btn.pack(side=tk.LEFT, padx=(0, 10))

        cancel_btn = tk.Button(button_frame, text="Cancelar", 
                              command=geo_window.destroy,
                              bg="#757575", fg="white", font=("Arial", 10, "bold"))
        cancel_btn.pack(side=tk.LEFT)

    def ejecutar_analisis_geoespacial_avanzado(self, window):
        """Ejecutar análisis geoespacial avanzado"""
        try:
            analysis_type = self.geo_analysis_type.get()
            num_clusters = self.num_clusters.get()
            include_stats = self.include_statistics.get()
            generate_heatmap = self.generate_heatmap.get()
            export_results = self.export_results.get()
            
            # Obtener variable de análisis si existe
            analysis_var = None
            if hasattr(self, 'analysis_variable'):
                analysis_var = self.analysis_variable.get()
            
            # Ejecutar según tipo de análisis
            if analysis_type == "provincias":
                results = self.analisis_geoespacial_por_provincias(analysis_var, include_stats, export_results)
            elif analysis_type == "clustering":
                results = self.analisis_clustering_espacial(num_clusters, analysis_var, include_stats)
            elif analysis_type == "densidad":
                results = self.analisis_densidad_espacial(analysis_var, generate_heatmap)
            else:  # completo
                results = self.analisis_geoespacial_completo(num_clusters, analysis_var, 
                                                           include_stats, generate_heatmap, export_results)
            
            # Mostrar resultados
            self.mostrar_resultados_geoespaciales(results, analysis_type)
            
            window.destroy()
            
        except Exception as e:
            logging.error(f"Error ejecutando análisis geoespacial: {e}")
            messagebox.showerror("Error", f"Error al ejecutar análisis geoespacial: {str(e)}")

    def analisis_geoespacial_por_provincias(self, analysis_var, include_stats, export_results):
        """Análisis geoespacial basado en provincias (sin coordenadas)"""
        if 'provincia' not in self.df.columns:
            raise ValueError("No se encontró la columna 'provincia' en los datos")
        
        results = {
            'tipo': 'provincias',
            'estadisticas': {},
            'visualizaciones': []
        }
        
        # Agrupar por provincia
        df_provincias = self.df.groupby('provincia').agg({
            col: ['sum', 'mean', 'count'] for col in self.df.select_dtypes(include=[np.number]).columns
        }).round(2)
        
        # Crear visualizaciones
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Distribución por provincia
        ax1 = axes[0, 0]
        provincia_counts = self.df['provincia'].value_counts()
        provincia_counts.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Número de Registros por Provincia')
        ax1.set_xlabel('Provincia')
        ax1.set_ylabel('Cantidad de Registros')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Variable de análisis por provincia (si existe)
        if analysis_var and analysis_var in self.df.columns:
            ax2 = axes[0, 1]
            provincia_var = self.df.groupby('provincia')[analysis_var].sum().sort_values(ascending=False)
            provincia_var.plot(kind='bar', ax=ax2, color='lightcoral')
            ax2.set_title(f'{analysis_var.title()} por Provincia')
            ax2.set_xlabel('Provincia')
            ax2.set_ylabel(analysis_var.title())
            ax2.tick_params(axis='x', rotation=45)
            
            results['estadisticas']['variable_principal'] = {
                'nombre': analysis_var,
                'total': provincia_var.sum(),
                'promedio': provincia_var.mean(),
                'provincia_max': provincia_var.idxmax(),
                'valor_max': provincia_var.max(),
                'provincia_min': provincia_var.idxmin(),
                'valor_min': provincia_var.min()
            }
        
        # 3. Diversidad de cultivos/productos por provincia (si hay columna 'cultivo' o similar)
        ax3 = axes[1, 0]
        if 'cultivo' in self.df.columns:
            diversidad = self.df.groupby('provincia')['cultivo'].nunique().sort_values(ascending=False)
            diversidad.plot(kind='bar', ax=ax3, color='lightgreen')
            ax3.set_title('Diversidad de Cultivos por Provincia')
            ax3.set_xlabel('Provincia')
            ax3.set_ylabel('Número de Cultivos Diferentes')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'Diversidad de cultivos\nno disponible\n(falta columna "cultivo")', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Diversidad de Cultivos')
        
        # 4. Ranking de provincias
        ax4 = axes[1, 1]
        if analysis_var and analysis_var in self.df.columns:
            # Crear ranking combinado (cantidad de registros + variable de análisis)
            ranking_data = pd.DataFrame({
                'registros': provincia_counts,
                'variable': provincia_var
            }).fillna(0)
            
            # Normalizar para crear ranking
            ranking_data['score'] = (ranking_data['registros'] / ranking_data['registros'].max() * 0.4 + 
                                   ranking_data['variable'] / ranking_data['variable'].max() * 0.6)
            
            top_provincias = ranking_data['score'].sort_values(ascending=False).head(10)
            top_provincias.plot(kind='barh', ax=ax4, color='gold')
            ax4.set_title('Ranking de Provincias (Score Combinado)')
            ax4.set_xlabel('Score (Registros + Variable)')
        else:
            provincia_counts.head(10).plot(kind='barh', ax=ax4, color='gold')
            ax4.set_title('Top 10 Provincias por Registros')
            ax4.set_xlabel('Cantidad de Registros')
        
        plt.tight_layout()
        
        # Guardar visualización
        geo_provincias_file = OUTPUT_DIR / "analisis_geoespacial_provincias.png"
        plt.savefig(geo_provincias_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        results['visualizaciones'].append(str(geo_provincias_file))
        
        # Estadísticas generales
        if include_stats:
            results['estadisticas']['general'] = {
                'total_provincias': len(provincia_counts),
                'total_registros': len(self.df),
                'promedio_registros_por_provincia': provincia_counts.mean(),
                'provincia_mas_registros': provincia_counts.idxmax(),
                'max_registros': provincia_counts.max(),
                'provincia_menos_registros': provincia_counts.idxmin(),
                'min_registros': provincia_counts.min()
            }
        
        # Exportar resultados
        if export_results:
            export_file = OUTPUT_DIR / "analisis_provincias_detallado.csv"
            df_provincias.to_csv(export_file)
            results['export_file'] = str(export_file)
        
        return results

    def analisis_clustering_espacial(self, num_clusters, analysis_var, include_stats):
        """Análisis de clustering espacial usando coordenadas"""
        if 'Latitude' not in self.df.columns or 'Longitude' not in self.df.columns:
            raise ValueError("Se requieren coordenadas (Latitude, Longitude) para clustering espacial")
        
        # Filtrar datos con coordenadas válidas
        df_coords = self.df.dropna(subset=['Latitude', 'Longitude']).copy()
        
        if len(df_coords) < num_clusters:
            raise ValueError(f"No hay suficientes puntos con coordenadas ({len(df_coords)}) para {num_clusters} clusters")
        
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        results = {
            'tipo': 'clustering',
            'num_clusters': num_clusters,
            'estadisticas': {},
            'visualizaciones': []
        }
        
        # Preparar datos para clustering
        coords = df_coords[['Latitude', 'Longitude']].values
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        # Aplicar K-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(coords_scaled)
        
        df_coords['Cluster'] = clusters
        
        # Crear visualizaciones
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Mapa de clusters
        ax1 = axes[0, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, num_clusters))
        for i in range(num_clusters):
            cluster_data = df_coords[df_coords['Cluster'] == i]
            ax1.scatter(cluster_data['Longitude'], cluster_data['Latitude'], 
                       c=[colors[i]], label=f'Cluster {i+1}', s=50, alpha=0.7)
        
        # Centros de clusters
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        ax1.scatter(centers[:, 1], centers[:, 0], c='red', marker='x', s=200, linewidths=3, label='Centros')
        
        ax1.set_xlabel('Longitud')
        ax1.set_ylabel('Latitud')
        ax1.set_title('Clustering Espacial de Ubicaciones')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribución de puntos por cluster
        ax2 = axes[0, 1]
        cluster_counts = df_coords['Cluster'].value_counts().sort_index()
        bars = ax2.bar(range(num_clusters), cluster_counts.values, color=colors)
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Número de Puntos')
        ax2.set_title('Distribución de Puntos por Cluster')
        ax2.set_xticks(range(num_clusters))
        ax2.set_xticklabels([f'C{i+1}' for i in range(num_clusters)])
        
        # Añadir valores en las barras
        for bar, count in zip(bars, cluster_counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{count}', ha='center', va='bottom')
        
        # 3. Variable de análisis por cluster (si existe)
        if analysis_var and analysis_var in df_coords.columns:
            ax3 = axes[1, 0]
            cluster_var = df_coords.groupby('Cluster')[analysis_var].mean()
            bars = ax3.bar(range(num_clusters), cluster_var.values, color=colors)
            ax3.set_xlabel('Cluster')
            ax3.set_ylabel(f'{analysis_var.title()} Promedio')
            ax3.set_title(f'{analysis_var.title()} Promedio por Cluster')
            ax3.set_xticks(range(num_clusters))
            ax3.set_xticklabels([f'C{i+1}' for i in range(num_clusters)])
            
            # Añadir valores en las barras
            for bar, value in zip(bars, cluster_var.values):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{value:.1f}', ha='center', va='bottom')
        
        # 4. Análisis de dispersión por cluster
        ax4 = axes[1, 1]
        dispersions = []
        for i in range(num_clusters):
            cluster_data = df_coords[df_coords['Cluster'] == i]
            if len(cluster_data) > 1:
                center = centers[i]
                distances = np.sqrt((cluster_data['Latitude'] - center[0])**2 + 
                                  (cluster_data['Longitude'] - center[1])**2)
                dispersions.append(distances.std())
            else:
                dispersions.append(0)
        
        bars = ax4.bar(range(num_clusters), dispersions, color=colors)
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('Dispersión Espacial')
        ax4.set_title('Dispersión Espacial por Cluster')
        ax4.set_xticks(range(num_clusters))
        ax4.set_xticklabels([f'C{i+1}' for i in range(num_clusters)])
        
        plt.tight_layout()
        
        # Guardar visualización
        clustering_file = OUTPUT_DIR / "analisis_clustering_espacial.png"
        plt.savefig(clustering_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        results['visualizaciones'].append(str(clustering_file))
        
        # Estadísticas del clustering
        if include_stats:
            results['estadisticas'] = {
                'inertia': kmeans.inertia_,
                'puntos_analizados': len(df_coords),
                'clusters_info': {}
            }
            
            for i in range(num_clusters):
                cluster_data = df_coords[df_coords['Cluster'] == i]
                results['estadisticas']['clusters_info'][f'cluster_{i+1}'] = {
                    'num_puntos': len(cluster_data),
                    'centro_lat': centers[i][0],
                    'centro_lon': centers[i][1],
                    'dispersion': dispersions[i]
                }
                
                if analysis_var and analysis_var in cluster_data.columns:
                    results['estadisticas']['clusters_info'][f'cluster_{i+1}'][f'{analysis_var}_promedio'] = cluster_data[analysis_var].mean()
        
        return results

    def analisis_densidad_espacial(self, analysis_var, generate_heatmap):
        """Análisis de densidad espacial"""
        if 'Latitude' not in self.df.columns or 'Longitude' not in self.df.columns:
            raise ValueError("Se requieren coordenadas (Latitude, Longitude) para análisis de densidad")
        
        df_coords = self.df.dropna(subset=['Latitude', 'Longitude']).copy()
        
        results = {
            'tipo': 'densidad',
            'estadisticas': {},
            'visualizaciones': []
        }
        
        # Crear grilla para análisis de densidad
        lat_min, lat_max = df_coords['Latitude'].min(), df_coords['Latitude'].max()
        lon_min, lon_max = df_coords['Longitude'].min(), df_coords['Longitude'].max()
        
        # Crear bins para la grilla
        lat_bins = np.linspace(lat_min, lat_max, 20)
        lon_bins = np.linspace(lon_min, lon_max, 20)
        
        # Calcular densidad
        density, _, _ = np.histogram2d(df_coords['Latitude'], df_coords['Longitude'], 
                                     bins=[lat_bins, lon_bins])
        
        # Crear visualizaciones
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Mapa de densidad
        ax1 = axes[0, 0]
        im1 = ax1.imshow(density, extent=[lon_min, lon_max, lat_min, lat_max], 
                        origin='lower', cmap='YlOrRd', aspect='auto')
        ax1.scatter(df_coords['Longitude'], df_coords['Latitude'], c='blue', s=10, alpha=0.5)
        ax1.set_xlabel('Longitud')
        ax1.set_ylabel('Latitud')
        ax1.set_title('Mapa de Densidad Espacial')
        plt.colorbar(im1, ax=ax1, label='Densidad')
        
        # 2. Histograma de latitudes
        ax2 = axes[0, 1]
        ax2.hist(df_coords['Latitude'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Latitud')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Latitudes')
        ax2.grid(True, alpha=0.3)
        
        # 3. Histograma de longitudes
        ax3 = axes[1, 0]
        ax3.hist(df_coords['Longitude'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.set_xlabel('Longitud')
        ax3.set_ylabel('Frecuencia')
        ax3.set_title('Distribución de Longitudes')
        ax3.grid(True, alpha=0.3)
        
        # 4. Análisis de variable por densidad (si existe)
        if analysis_var and analysis_var in df_coords.columns:
            ax4 = axes[1, 1]
            
            # Crear bins de densidad
            df_coords['density_bin'] = pd.cut(df_coords.groupby(['Latitude', 'Longitude']).size(), 
                                            bins=5, labels=['Muy Baja', 'Baja', 'Media', 'Alta', 'Muy Alta'])
            
            density_var = df_coords.groupby('density_bin')[analysis_var].mean()
            bars = ax4.bar(range(len(density_var)), density_var.values, 
                          color=['lightblue', 'yellow', 'orange', 'red', 'darkred'])
            ax4.set_xlabel('Nivel de Densidad')
            ax4.set_ylabel(f'{analysis_var.title()} Promedio')
            ax4.set_title(f'{analysis_var.title()} por Nivel de Densidad')
            ax4.set_xticks(range(len(density_var)))
            ax4.set_xticklabels(density_var.index, rotation=45)
            
            # Añadir valores en las barras
            for bar, value in zip(bars, density_var.values):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{value:.1f}', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, f'Análisis de variable\nno disponible\n(seleccione una variable numérica)', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Análisis por Variable')
        
        plt.tight_layout()
        
        # Guardar visualización
        density_file = OUTPUT_DIR / "analisis_densidad_espacial.png"
        plt.savefig(density_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        results['visualizaciones'].append(str(density_file))
        
        # Estadísticas de densidad
        results['estadisticas'] = {
            'puntos_analizados': len(df_coords),
            'area_cobertura': {
                'lat_min': lat_min,
                'lat_max': lat_max,
                'lon_min': lon_min,
                'lon_max': lon_max,
                'rango_lat': lat_max - lat_min,
                'rango_lon': lon_max - lon_min
            },
            'densidad_maxima': density.max(),
            'densidad_promedio': density.mean()
        }
        
        return results

    def analisis_geoespacial_completo(self, num_clusters, analysis_var, include_stats, generate_heatmap, export_results):
        """Análisis geoespacial completo combinando todos los métodos"""
        results = {
            'tipo': 'completo',
            'componentes': [],
            'estadisticas': {},
            'visualizaciones': []
        }
        
        # Ejecutar análisis por provincias
        try:
            prov_results = self.analisis_geoespacial_por_provincias(analysis_var, include_stats, export_results)
            results['componentes'].append('provincias')
            results['visualizaciones'].extend(prov_results['visualizaciones'])
            if 'estadisticas' in prov_results:
                results['estadisticas']['provincias'] = prov_results['estadisticas']
        except Exception as e:
            logging.warning(f"No se pudo completar análisis por provincias: {e}")
        
        # Ejecutar clustering si hay coordenadas
        if 'Latitude' in self.df.columns and 'Longitude' in self.df.columns:
            try:
                cluster_results = self.analisis_clustering_espacial(num_clusters, analysis_var, include_stats)
                results['componentes'].append('clustering')
                results['visualizaciones'].extend(cluster_results['visualizaciones'])
                if 'estadisticas' in cluster_results:
                    results['estadisticas']['clustering'] = cluster_results['estadisticas']
            except Exception as e:
                logging.warning(f"No se pudo completar clustering espacial: {e}")
            
            # Ejecutar análisis de densidad si hay coordenadas
            try:
                density_results = self.analisis_densidad_espacial(analysis_var, generate_heatmap)
                results['componentes'].append('densidad')
                results['visualizaciones'].extend(density_results['visualizaciones'])
                if 'estadisticas' in density_results:
                    results['estadisticas']['densidad'] = density_results['estadisticas']
            except Exception as e:
                logging.warning(f"No se pudo completar análisis de densidad: {e}")
        
        return results

    def mostrar_resultados_geoespaciales(self, results, analysis_type):
        """Mostrar ventana con resultados del análisis geoespacial"""
        results_window = tk.Toplevel(self.root)
        results_window.title("Resultados del Análisis Geoespacial")
        results_window.geometry("700x500")
        results_window.resizable(True, True)
        
        # Frame principal con scrollbar
        main_frame = tk.Frame(results_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Canvas y scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Título
        title_label = tk.Label(scrollable_frame, text="Análisis Geoespacial - Resultados", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Resumen del análisis
        summary_frame = tk.LabelFrame(scrollable_frame, text="Resumen del Análisis", 
                                     font=("Arial", 12, "bold"))
        summary_frame.pack(fill=tk.X, pady=(0, 15))
        
        summary_text = f"ANÁLISIS GEOESPACIAL COMPLETADO\n\n"
        summary_text += f"Tipo de análisis: {analysis_type.upper()}\n"
        
        if 'componentes' in results:
            summary_text += f"Componentes ejecutados: {', '.join(results['componentes'])}\n"
        
        summary_text += f"Visualizaciones generadas: {len(results.get('visualizaciones', []))}\n"
        
        summary_label = tk.Label(summary_frame, text=summary_text, font=("Arial", 10), 
                                justify=tk.LEFT, bg="lightblue")
        summary_label.pack(anchor="w", padx=10, pady=10, fill=tk.X)
        
        # Estadísticas detalladas por componente
        if 'estadisticas' in results:
            for componente, stats in results['estadisticas'].items():
                stats_frame = tk.LabelFrame(scrollable_frame, 
                                          text=f"Estadísticas - {componente.title()}", 
                                          font=("Arial", 11, "bold"))
                stats_frame.pack(fill=tk.X, pady=(0, 10))
                
                # Mostrar estadísticas según el componente
                if componente == 'provincias' and 'general' in stats:
                    stats_text = f"Total provincias: {stats['general']['total_provincias']}\n"
                    stats_text += f"Total registros: {stats['general']['total_registros']}\n"
                    stats_text += f"Provincia con más registros: {stats['general']['provincia_mas_registros']}\n"
                    
                    if 'variable_principal' in stats:
                        var_info = stats['variable_principal']
                        stats_text += f"\nVariable analizada: {var_info['nombre']}\n"
                        stats_text += f"Provincia líder: {var_info['provincia_max']} ({var_info['valor_max']:.2f})\n"
                
                elif componente == 'clustering':
                    stats_text = f"Número de clusters: {results.get('num_clusters', 'N/A')}\n"
                    stats_text += f"Puntos analizados: {stats.get('puntos_analizados', 0)}\n"
                    stats_text += f"Inercia del modelo: {stats.get('inertia', 0):.2f}\n"
                
                elif componente == 'densidad':
                    stats_text = f"Puntos analizados: {stats.get('puntos_analizados', 0)}\n"
                    if 'area_cobertura' in stats:
                        area = stats['area_cobertura']
                        stats_text += f"Rango latitud: {area['rango_lat']:.2f}°\n"
                        stats_text += f"Rango longitud: {area['rango_lon']:.2f}°\n"
                    stats_text += f"Densidad máxima: {stats.get('densidad_maxima', 0):.2f}\n"
                
                else:
                    stats_text = "Estadísticas disponibles en los archivos exportados."
                
                stats_label = tk.Label(stats_frame, text=stats_text, font=("Arial", 9), 
                                     justify=tk.LEFT, bg="white")
                stats_label.pack(anchor="w", padx=10, pady=10, fill=tk.X)
        
        # Lista de archivos generados
        files_frame = tk.LabelFrame(scrollable_frame, text="Archivos Generados", 
                                   font=("Arial", 12, "bold"))
        files_frame.pack(fill=tk.X, pady=(15, 0))
        
        files_text = "VISUALIZACIONES GUARDADAS:\n\n"
        for i, file_path in enumerate(results.get('visualizaciones', []), 1):
            file_name = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
            files_text += f"{i}. {file_name}\n"
        
        if 'export_file' in results:
            files_text += f"\nARCHIVO DE DATOS EXPORTADO:\n"
            export_name = results['export_file'].split('/')[-1] if '/' in results['export_file'] else results['export_file'].split('\\')[-1]
            files_text += f"• {export_name}\n"
        
        files_text += f"\nTodos los archivos se encuentran en la carpeta 'output'."
        
        files_label = tk.Label(files_frame, text=files_text, font=("Arial", 9), 
                              justify=tk.LEFT, bg="lightyellow")
        files_label.pack(anchor="w", padx=10, pady=10, fill=tk.X)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Botón cerrar
        close_btn = tk.Button(scrollable_frame, text="Cerrar Resultados", command=results_window.destroy,
                             bg="#FF9800", fg="white", font=("Arial", 12, "bold"))
        close_btn.pack(pady=(20, 0))
        
        # Mensaje de éxito
        messagebox.showinfo("Análisis Geoespacial Completado", 
                           f"Análisis geoespacial {analysis_type} completado exitosamente.\n"
                           f"Componentes ejecutados: {len(results.get('componentes', []))}\n"
                           f"Visualizaciones generadas: {len(results.get('visualizaciones', []))}\n"
                           f"Archivos guardados en la carpeta 'output'.")

    def analisis_correlacion(self):
        if self.df.empty or not any(self.df.select_dtypes(include=[float, int]).columns):
            messagebox.showwarning("Advertencia", "Por favor, asegúrese de que el archivo CSV contenga columnas numéricas.")
            return

        numeric_columns = self.df.select_dtypes(include=[float, int])
        
        # Check if there are enough numeric columns to calculate correlation
        if numeric_columns.shape[1] < 2:
            messagebox.showwarning("Advertencia", "El archivo CSV no contiene suficientes columnas numéricas para calcular la correlación.")
            return

        # Drop rows with any missing values in numeric columns
        numeric_columns = numeric_columns.dropna()

        correlacion = numeric_columns.corr()

        sns.heatmap(correlacion, annot=True, cmap='coolwarm', linewidths=.5)
        plt.title('Matriz de Correlación')
        plt.tight_layout()

        analisis_correlacion_file = OUTPUT_DIR / "analisis_correlacion.png"
        plt.savefig(analisis_correlacion_file)
        plt.show()
        logging.debug(f"Gráfica de análisis de correlación guardada en {analisis_correlacion_file}")

        messagebox.showinfo("Análisis de Correlación", f"Gráfica de análisis de correlación guardada en {analisis_correlacion_file}")

    def modelos_predictivos(self):
        if self.df.empty:
            messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return

        try:
            # Obtener columnas numéricas
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) < 2:
                messagebox.showwarning("Advertencia", "Se necesitan al menos 2 columnas numéricas para crear modelos predictivos.")
                return

            # Crear ventana de selección de variables
            self.crear_ventana_modelos_predictivos(numeric_columns)
            
        except Exception as e:
            logging.error(f"Error en modelos predictivos: {e}")
            messagebox.showerror("Error", f"Error al ejecutar modelos predictivos: {str(e)}")

    def crear_ventana_modelos_predictivos(self, numeric_columns):
        """Crear ventana para seleccionar variables y ejecutar modelos"""
        model_window = tk.Toplevel(self.root)
        model_window.title("Modelos Predictivos")
        model_window.geometry("500x400")
        model_window.resizable(True, True)

        # Frame principal
        main_frame = tk.Frame(model_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título
        title_label = tk.Label(main_frame, text="Configuración de Modelos Predictivos", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))

        # Selección de variable objetivo
        target_frame = tk.LabelFrame(main_frame, text="Variable Objetivo (Y)", font=("Arial", 10, "bold"))
        target_frame.pack(fill=tk.X, pady=(0, 10))

        self.target_var = tk.StringVar()
        target_combo = ttk.Combobox(target_frame, textvariable=self.target_var, values=numeric_columns, state="readonly")
        target_combo.pack(pady=10, padx=10, fill=tk.X)
        target_combo.current(0)

        # Selección de variables predictoras
        features_frame = tk.LabelFrame(main_frame, text="Variables Predictoras (X)", font=("Arial", 10, "bold"))
        features_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Frame con scrollbar para las variables
        canvas = tk.Canvas(features_frame)
        scrollbar = ttk.Scrollbar(features_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Variables de control para checkboxes
        self.feature_vars = {}
        for col in numeric_columns:
            var = tk.BooleanVar(value=True)
            self.feature_vars[col] = var
            cb = tk.Checkbutton(scrollable_frame, text=col, variable=var)
            cb.pack(anchor="w", padx=10, pady=2)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Selección de modelo
        model_frame = tk.LabelFrame(main_frame, text="Tipo de Modelo", font=("Arial", 10, "bold"))
        model_frame.pack(fill=tk.X, pady=(0, 10))

        self.model_type = tk.StringVar(value="linear")
        
        linear_rb = tk.Radiobutton(model_frame, text="Regresión Lineal", variable=self.model_type, value="linear")
        linear_rb.pack(anchor="w", padx=10, pady=2)
        
        forest_rb = tk.Radiobutton(model_frame, text="Random Forest", variable=self.model_type, value="forest")
        forest_rb.pack(anchor="w", padx=10, pady=2)
        
        both_rb = tk.Radiobutton(model_frame, text="Ambos Modelos", variable=self.model_type, value="both")
        both_rb.pack(anchor="w", padx=10, pady=2)

        # Botones
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        execute_btn = tk.Button(button_frame, text="Ejecutar Modelos", 
                               command=lambda: self.ejecutar_modelos_predictivos(model_window),
                               bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        execute_btn.pack(side=tk.LEFT, padx=(0, 10))

        cancel_btn = tk.Button(button_frame, text="Cancelar", 
                              command=model_window.destroy,
                              bg="#f44336", fg="white", font=("Arial", 10, "bold"))
        cancel_btn.pack(side=tk.LEFT)

    def ejecutar_modelos_predictivos(self, window):
        """Ejecutar los modelos predictivos seleccionados"""
        try:
            # Obtener variable objetivo
            target_column = self.target_var.get()
            
            # Obtener variables predictoras seleccionadas (excluyendo la variable objetivo)
            selected_features = [col for col, var in self.feature_vars.items() if var.get() and col != target_column]
            
            if len(selected_features) == 0:
                messagebox.showwarning("Advertencia", f"Debe seleccionar al menos una variable predictora diferente a la variable objetivo '{target_column}'.")
                return
            
            # Mostrar información de lo que se va a procesar
            info_msg = f"PROCESANDO MODELOS PREDICTIVOS\n\n"
            info_msg += f"Variable Objetivo: {target_column}\n"
            info_msg += f"Variables Predictoras: {', '.join(selected_features)}\n\n"
            info_msg += f"Datos disponibles: {len(self.df)} filas\n"
            info_msg += "Iniciando procesamiento..."
            
            # Mostrar mensaje informativo
            messagebox.showinfo("Procesando", info_msg)
            
            # Preparar datos
            df_clean = self.df[selected_features + [target_column]].dropna()
            
            if df_clean.empty:
                messagebox.showwarning("Advertencia", "No hay datos válidos después de eliminar valores faltantes.")
                return
            
            X = df_clean[selected_features]
            y = df_clean[target_column]
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Escalar datos
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model_type = self.model_type.get()
            results = {}
            
            # Ejecutar modelos según selección
            if model_type in ["linear", "both"]:
                results["Linear"] = self.entrenar_modelo_lineal(X_train_scaled, X_test_scaled, y_train, y_test)
            
            if model_type in ["forest", "both"]:
                results["Random Forest"] = self.entrenar_random_forest(X_train, X_test, y_train, y_test)
            
            # Mostrar resultados y generar gráficos
            self.mostrar_resultados_modelos(results, target_column, selected_features)
            
            window.destroy()
            
        except Exception as e:
            logging.error(f"Error ejecutando modelos: {e}")
            messagebox.showerror("Error", f"Error al ejecutar modelos: {str(e)}")

    def entrenar_modelo_lineal(self, X_train, X_test, y_train, y_test):
        """Entrenar modelo de regresión lineal"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        return {
            'model': model,
            'predictions': y_pred,
            'actual': y_test,
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }

    def entrenar_random_forest(self, X_train, X_test, y_train, y_test):
        """Entrenar modelo Random Forest"""
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        return {
            'model': model,
            'predictions': y_pred,
            'actual': y_test,
            'mse': mse,
            'r2': r2,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'feature_importance': model.feature_importances_
        }

    def mostrar_resultados_modelos(self, results, target_column, feature_columns):
        """Mostrar resultados de los modelos y generar gráficos"""
        
        # Crear figura con subplots
        n_models = len(results)
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        model_names = list(results.keys())
        
        # Generar gráficos para cada modelo
        for i, (model_name, result) in enumerate(results.items()):
            
            # Gráfico de predicciones vs valores reales
            axes[0, i].scatter(result['actual'], result['predictions'], alpha=0.6)
            axes[0, i].plot([result['actual'].min(), result['actual'].max()], 
                           [result['actual'].min(), result['actual'].max()], 'r--', lw=2)
            axes[0, i].set_xlabel('Valores Reales')
            axes[0, i].set_ylabel('Predicciones')
            axes[0, i].set_title(f'{model_name}\nR² = {result["r2"]:.3f}')
            axes[0, i].grid(True, alpha=0.3)
            
            # Gráfico de residuos
            residuos = result['actual'] - result['predictions']
            axes[1, i].scatter(result['predictions'], residuos, alpha=0.6)
            axes[1, i].axhline(y=0, color='r', linestyle='--')
            axes[1, i].set_xlabel('Predicciones')
            axes[1, i].set_ylabel('Residuos')
            axes[1, i].set_title(f'Residuos - {model_name}')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico
        modelos_file = OUTPUT_DIR / "modelos_predictivos.png"
        plt.savefig(modelos_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generar gráfico de importancia de características (si hay Random Forest)
        if 'Random Forest' in results and 'feature_importance' in results['Random Forest']:
            self.grafico_importancia_caracteristicas(results['Random Forest']['feature_importance'], 
                                                   feature_columns)
        
        # Mostrar métricas en ventana de resultados
        self.mostrar_ventana_resultados(results, target_column, feature_columns)
        
        logging.debug(f"Gráficos de modelos predictivos guardados en {modelos_file}")

    def grafico_importancia_caracteristicas(self, importances, feature_names):
        """Generar gráfico de importancia de características"""
        plt.figure(figsize=(10, 6))
        
        # Ordenar por importancia
        indices = np.argsort(importances)[::-1]
        
        plt.bar(range(len(importances)), importances[indices])
        plt.title('Importancia de Características - Random Forest')
        plt.xlabel('Características')
        plt.ylabel('Importancia')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        
        # Guardar gráfico
        importancia_file = OUTPUT_DIR / "importancia_caracteristicas.png"
        plt.savefig(importancia_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logging.debug(f"Gráfico de importancia guardado en {importancia_file}")

    def mostrar_ventana_resultados(self, results, target_column, feature_columns):
        """Mostrar ventana con métricas detalladas"""
        results_window = tk.Toplevel(self.root)
        results_window.title("Resultados de Modelos Predictivos")
        results_window.geometry("600x500")
        
        # Frame principal con scrollbar
        main_frame = tk.Frame(results_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        title_label = tk.Label(main_frame, text="Resultados de Modelos Predictivos", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Información del modelo
        info_text = f"Variable Objetivo: {target_column}\n"
        info_text += f"Variables Predictoras: {', '.join(feature_columns)}\n\n"
        
        info_label = tk.Label(main_frame, text=info_text, font=("Arial", 10), justify=tk.LEFT)
        info_label.pack(anchor="w", pady=(0, 10))
        
        # Resultados por modelo
        for model_name, result in results.items():
            model_frame = tk.LabelFrame(main_frame, text=f"Resultados - {model_name}", 
                                       font=("Arial", 11, "bold"))
            model_frame.pack(fill=tk.X, pady=(0, 10))
            
            metrics_text = f"R² Score: {result['r2']:.4f}\n"
            metrics_text += f"Error Cuadrático Medio (MSE): {result['mse']:.4f}\n"
            metrics_text += f"Raíz del Error Cuadrático Medio (RMSE): {result['rmse']:.4f}\n"
            metrics_text += f"Error Absoluto Medio (MAE): {result['mae']:.4f}"
            
            metrics_label = tk.Label(model_frame, text=metrics_text, font=("Arial", 9), 
                                   justify=tk.LEFT, bg="white")
            metrics_label.pack(anchor="w", padx=10, pady=10, fill=tk.X)
        
        # Botón cerrar
        close_btn = tk.Button(main_frame, text="Cerrar", command=results_window.destroy,
                             bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        close_btn.pack(pady=(20, 0))
        
        # Mensaje de éxito
        messagebox.showinfo("Modelos Predictivos", 
                           "Modelos predictivos ejecutados exitosamente. "
                           "Gráficos guardados en la carpeta 'output'.")

    def analisis_rapido(self):
        """Función para realizar un análisis rápido con variables objetivo y predictoras automáticas"""
        if self.df.empty:
            messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return

        try:
            # Obtener columnas numéricas
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) < 2:
                messagebox.showwarning("Advertencia", "Se necesitan al menos 2 columnas numéricas para el análisis rápido.")
                return

            # Seleccionar automáticamente la primera columna como objetivo y el resto como predictoras
            target_column = numeric_columns[0]
            feature_columns = numeric_columns[1:]
            
            # Mostrar información de las variables seleccionadas
            info_msg = f"ANÁLISIS RÁPIDO AUTOMÁTICO\n\n"
            info_msg += f"Variable Objetivo: {target_column}\n"
            info_msg += f"Variables Predictoras: {', '.join(feature_columns)}\n\n"
            info_msg += "¿Desea continuar con este análisis?"
            
            result = messagebox.askyesno("Análisis Rápido", info_msg)
            if not result:
                return
            
            # Preparar datos
            df_clean = self.df[feature_columns + [target_column]].dropna()
            
            if df_clean.empty:
                messagebox.showwarning("Advertencia", "No hay datos válidos después de eliminar valores faltantes.")
                return
            
            X = df_clean[feature_columns]
            y = df_clean[target_column]
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Ejecutar ambos modelos
            results = {}
            
            # Modelo lineal (con escalado)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            results["Linear"] = self.entrenar_modelo_lineal(X_train_scaled, X_test_scaled, y_train, y_test)
            
            # Random Forest (sin escalado)
            results["Random Forest"] = self.entrenar_random_forest(X_train, X_test, y_train, y_test)
            
            # Mostrar resultados
            self.mostrar_resultados_modelos(results, target_column, feature_columns)
            
        except Exception as e:
            logging.error(f"Error en análisis rápido: {e}")
            messagebox.showerror("Error", f"Error al ejecutar análisis rápido: {str(e)}")

    def analisis_riesgos(self):
        if self.df.empty:
            messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return

        try:
            # Crear ventana de análisis de riesgos
            self.crear_ventana_analisis_riesgos()
            
        except Exception as e:
            logging.error(f"Error en análisis de riesgos: {e}")
            messagebox.showerror("Error", f"Error al ejecutar análisis de riesgos: {str(e)}")

    def crear_ventana_analisis_riesgos(self):
        """Crear ventana para configurar y ejecutar análisis de riesgos"""
        risk_window = tk.Toplevel(self.root)
        risk_window.title("Análisis de Riesgos Avanzado")
        risk_window.geometry("600x500")
        risk_window.resizable(True, True)

        # Frame principal
        main_frame = tk.Frame(risk_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Título
        title_label = tk.Label(main_frame, text="Análisis de Riesgos del Proyecto", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))

        # Descripción
        desc_text = ("Este análisis evalúa riesgos técnicos, operacionales, de datos y de negocio\n"
                    "específicos para aplicaciones de análisis de datos agrícolas.")
        desc_label = tk.Label(main_frame, text=desc_text, font=("Arial", 10), 
                             justify=tk.CENTER, fg="gray")
        desc_label.pack(pady=(0, 20))

        # Selección de tipo de análisis
        analysis_frame = tk.LabelFrame(main_frame, text="Tipo de Análisis", font=("Arial", 10, "bold"))
        analysis_frame.pack(fill=tk.X, pady=(0, 10))

        self.risk_analysis_type = tk.StringVar(value="completo")
        
        completo_rb = tk.Radiobutton(analysis_frame, text="Análisis Completo (Recomendado)", 
                                    variable=self.risk_analysis_type, value="completo")
        completo_rb.pack(anchor="w", padx=10, pady=5)
        
        tecnico_rb = tk.Radiobutton(analysis_frame, text="Solo Riesgos Técnicos", 
                                   variable=self.risk_analysis_type, value="tecnico")
        tecnico_rb.pack(anchor="w", padx=10, pady=2)
        
        datos_rb = tk.Radiobutton(analysis_frame, text="Solo Riesgos de Datos", 
                                 variable=self.risk_analysis_type, value="datos")
        datos_rb.pack(anchor="w", padx=10, pady=2)
        
        operacional_rb = tk.Radiobutton(analysis_frame, text="Solo Riesgos Operacionales", 
                                       variable=self.risk_analysis_type, value="operacional")
        operacional_rb.pack(anchor="w", padx=10, pady=2)

        # Configuración de severidad
        severity_frame = tk.LabelFrame(main_frame, text="Configuración de Evaluación", font=("Arial", 10, "bold"))
        severity_frame.pack(fill=tk.X, pady=(0, 10))

        self.include_mitigation = tk.BooleanVar(value=True)
        mitigation_cb = tk.Checkbutton(severity_frame, text="Incluir estrategias de mitigación", 
                                      variable=self.include_mitigation)
        mitigation_cb.pack(anchor="w", padx=10, pady=5)

        self.detailed_metrics = tk.BooleanVar(value=True)
        metrics_cb = tk.Checkbutton(severity_frame, text="Incluir métricas detalladas", 
                                   variable=self.detailed_metrics)
        metrics_cb.pack(anchor="w", padx=10, pady=2)

        self.data_quality_check = tk.BooleanVar(value=True)
        quality_cb = tk.Checkbutton(severity_frame, text="Evaluar calidad de datos cargados", 
                                   variable=self.data_quality_check)
        quality_cb.pack(anchor="w", padx=10, pady=2)

        # Botones
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))

        execute_btn = tk.Button(button_frame, text="Ejecutar Análisis de Riesgos", 
                               command=lambda: self.ejecutar_analisis_riesgos_avanzado(risk_window),
                               bg="#FF5722", fg="white", font=("Arial", 11, "bold"))
        execute_btn.pack(side=tk.LEFT, padx=(0, 10))

        cancel_btn = tk.Button(button_frame, text="Cancelar", 
                              command=risk_window.destroy,
                              bg="#757575", fg="white", font=("Arial", 10, "bold"))
        cancel_btn.pack(side=tk.LEFT)

    def ejecutar_analisis_riesgos_avanzado(self, window):
        """Ejecutar análisis de riesgos avanzado"""
        try:
            analysis_type = self.risk_analysis_type.get()
            include_mitigation = self.include_mitigation.get()
            detailed_metrics = self.detailed_metrics.get()
            data_quality_check = self.data_quality_check.get()
            
            # Definir riesgos por categoría
            riesgos_completos = self.definir_riesgos_proyecto()
            
            # Filtrar según tipo de análisis
            if analysis_type == "tecnico":
                riesgos_filtrados = {k: v for k, v in riesgos_completos.items() if v['categoria'] == 'Técnico'}
            elif analysis_type == "datos":
                riesgos_filtrados = {k: v for k, v in riesgos_completos.items() if v['categoria'] == 'Datos'}
            elif analysis_type == "operacional":
                riesgos_filtrados = {k: v for k, v in riesgos_completos.items() if v['categoria'] == 'Operacional'}
            else:
                riesgos_filtrados = riesgos_completos
            
            # Evaluar calidad de datos si está habilitado
            data_quality_score = 0
            if data_quality_check and not self.df.empty:
                data_quality_score = self.evaluar_calidad_datos()
                # Ajustar probabilidades basado en calidad de datos
                riesgos_filtrados = self.ajustar_riesgos_por_calidad_datos(riesgos_filtrados, data_quality_score)
            
            # Generar visualizaciones
            self.generar_visualizaciones_riesgos(riesgos_filtrados, analysis_type, data_quality_score)
            
            # Mostrar resultados detallados
            self.mostrar_resultados_riesgos(riesgos_filtrados, include_mitigation, detailed_metrics, 
                                           data_quality_score, analysis_type)
            
            window.destroy()
            
        except Exception as e:
            logging.error(f"Error ejecutando análisis de riesgos: {e}")
            messagebox.showerror("Error", f"Error al ejecutar análisis de riesgos: {str(e)}")

    def definir_riesgos_proyecto(self):
        """Definir riesgos específicos del proyecto de análisis de datos agrícolas"""
        return {
            "Dependencias_Externas": {
                "categoria": "Técnico",
                "descripcion": "Fallas en APIs de geocodificación (Nominatim) o librerías críticas",
                "probabilidad": 0.25,
                "impacto": 0.8,
                "severidad": "Alta",
                "mitigacion": [
                    "Implementar cache local para geocodificación",
                    "Usar múltiples proveedores de geocodificación",
                    "Versionar dependencias específicas",
                    "Implementar fallbacks para funcionalidades críticas"
                ]
            },
            "Calidad_Datos_Agricolas": {
                "categoria": "Datos",
                "descripcion": "Datos agrícolas incompletos, inconsistentes o con errores estacionales",
                "probabilidad": 0.6,
                "impacto": 0.9,
                "severidad": "Crítica",
                "mitigacion": [
                    "Validación automática de rangos de datos agrícolas",
                    "Detección de outliers estacionales",
                    "Imputación inteligente basada en patrones históricos",
                    "Alertas de calidad de datos en tiempo real"
                ]
            },
            "Escalabilidad_Datasets": {
                "categoria": "Técnico",
                "descripcion": "Rendimiento degradado con datasets grandes (>100MB) o muchas provincias",
                "probabilidad": 0.4,
                "impacto": 0.7,
                "severidad": "Media",
                "mitigacion": [
                    "Implementar procesamiento por chunks",
                    "Optimizar consultas con pandas",
                    "Usar lazy loading para visualizaciones",
                    "Implementar progress bars para operaciones largas"
                ]
            },
            "Precision_Modelos_ML": {
                "categoria": "Técnico",
                "descripcion": "Modelos predictivos con baja precisión debido a variables climáticas no consideradas",
                "probabilidad": 0.5,
                "impacto": 0.8,
                "severidad": "Alta",
                "mitigacion": [
                    "Incorporar datos climáticos históricos",
                    "Validación cruzada temporal específica",
                    "Ensemble de múltiples modelos",
                    "Métricas de evaluación específicas para agricultura"
                ]
            },
            "Geocodificacion_Inexacta": {
                "categoria": "Datos",
                "descripcion": "Coordenadas incorrectas por nombres ambiguos de departamentos/provincias",
                "probabilidad": 0.35,
                "impacto": 0.6,
                "severidad": "Media",
                "mitigacion": [
                    "Validación manual de coordenadas críticas",
                    "Base de datos local de ubicaciones argentinas",
                    "Algoritmo de verificación de coherencia geográfica",
                    "Interface para corrección manual de coordenadas"
                ]
            },
            "Compatibilidad_SO": {
                "categoria": "Técnico",
                "descripcion": "Problemas de compatibilidad entre diferentes sistemas operativos",
                "probabilidad": 0.3,
                "impacto": 0.5,
                "severidad": "Baja",
                "mitigacion": [
                    "Testing en múltiples plataformas",
                    "Uso de paths relativos consistentes",
                    "Containerización con Docker",
                    "Documentación específica por SO"
                ]
            },
            "Perdida_Datos_Temporales": {
                "categoria": "Operacional",
                "descripcion": "Pérdida de análisis temporal por datos de campañas faltantes",
                "probabilidad": 0.4,
                "impacto": 0.7,
                "severidad": "Media",
                "mitigacion": [
                    "Backup automático de datasets procesados",
                    "Versionado de datos por campaña",
                    "Interpolación inteligente para campañas faltantes",
                    "Alertas de continuidad temporal"
                ]
            },
            "Usabilidad_Interface": {
                "categoria": "Operacional",
                "descripcion": "Interface compleja para usuarios no técnicos del sector agrícola",
                "probabilidad": 0.6,
                "impacto": 0.6,
                "severidad": "Media",
                "mitigacion": [
                    "Wizards paso a paso para análisis complejos",
                    "Tooltips explicativos con terminología agrícola",
                    "Presets para análisis comunes",
                    "Manual de usuario específico para el sector"
                ]
            },
            "Seguridad_Datos": {
                "categoria": "Operacional",
                "descripcion": "Exposición de datos sensibles de producción agrícola",
                "probabilidad": 0.2,
                "impacto": 0.9,
                "severidad": "Alta",
                "mitigacion": [
                    "Encriptación de archivos sensibles",
                    "Anonimización de datos de productores",
                    "Control de acceso por roles",
                    "Auditoría de acceso a datos"
                ]
            },
            "Interpretacion_Resultados": {
                "categoria": "Operacional",
                "descripcion": "Malinterpretación de resultados estadísticos por usuarios finales",
                "probabilidad": 0.7,
                "impacto": 0.8,
                "severidad": "Alta",
                "mitigacion": [
                    "Explicaciones automáticas de métricas",
                    "Visualizaciones intuitivas con contexto",
                    "Alertas sobre limitaciones de los análisis",
                    "Reportes ejecutivos con recomendaciones claras"
                ]
            }
        }

    def evaluar_calidad_datos(self):
        """Evaluar la calidad de los datos cargados"""
        if self.df.empty:
            return 0
        
        score = 0
        max_score = 100
        
        # Completitud de datos (30 puntos)
        completitud = (1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 30
        score += completitud
        
        # Consistencia de tipos de datos (20 puntos)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Verificar valores negativos en columnas que no deberían tenerlos
            cols_positivas = ['sup_sembrada', 'sup_cosechada', 'produccion', 'rendimiento']
            cols_check = [col for col in cols_positivas if col in numeric_cols]
            if cols_check:
                valores_negativos = sum((self.df[col] < 0).sum() for col in cols_check)
                consistencia = max(0, 20 - (valores_negativos / len(self.df)) * 20)
                score += consistencia
            else:
                score += 15  # Puntuación parcial si no hay columnas específicas
        
        # Coherencia temporal (25 puntos)
        if 'campaña' in self.df.columns:
            try:
                años = self.df['campaña'].str.split('/').str[0].astype(int)
                rango_años = años.max() - años.min()
                if rango_años > 0 and rango_años < 100:  # Rango razonable
                    coherencia_temporal = min(25, rango_años / 20 * 25)
                    score += coherencia_temporal
            except:
                score += 10  # Puntuación parcial si hay problemas de formato
        
        # Diversidad geográfica (25 puntos)
        if 'provincia' in self.df.columns:
            provincias_unicas = self.df['provincia'].nunique()
            diversidad = min(25, provincias_unicas / 24 * 25)  # Argentina tiene 24 provincias
            score += diversidad
        
        return min(score, max_score)

    def ajustar_riesgos_por_calidad_datos(self, riesgos, calidad_score):
        """Ajustar probabilidades de riesgos basado en la calidad de datos"""
        factor_ajuste = 1.0
        
        if calidad_score < 30:
            factor_ajuste = 1.5  # Aumentar riesgos si calidad es muy baja
        elif calidad_score < 60:
            factor_ajuste = 1.2  # Aumentar ligeramente
        elif calidad_score > 80:
            factor_ajuste = 0.8  # Reducir riesgos si calidad es alta
        
        # Ajustar solo riesgos relacionados con datos
        riesgos_datos = ['Calidad_Datos_Agricolas', 'Geocodificacion_Inexacta', 'Perdida_Datos_Temporales']
        
        for riesgo_key in riesgos_datos:
            if riesgo_key in riesgos:
                nueva_prob = min(0.95, riesgos[riesgo_key]['probabilidad'] * factor_ajuste)
                riesgos[riesgo_key]['probabilidad'] = nueva_prob
                
                # Actualizar severidad basada en nueva probabilidad
                if nueva_prob > 0.7:
                    riesgos[riesgo_key]['severidad'] = 'Crítica'
                elif nueva_prob > 0.5:
                    riesgos[riesgo_key]['severidad'] = 'Alta'
                elif nueva_prob > 0.3:
                    riesgos[riesgo_key]['severidad'] = 'Media'
                else:
                    riesgos[riesgo_key]['severidad'] = 'Baja'
        
        return riesgos

    def generar_visualizaciones_riesgos(self, riesgos, analysis_type, data_quality_score):
        """Generar visualizaciones avanzadas de riesgos"""
        
        # Preparar datos para visualización
        nombres = list(riesgos.keys())
        probabilidades = [riesgos[r]['probabilidad'] for r in nombres]
        impactos = [riesgos[r]['impacto'] for r in nombres]
        categorias = [riesgos[r]['categoria'] for r in nombres]
        severidades = [riesgos[r]['severidad'] for r in nombres]
        
        # Calcular riesgo total (probabilidad * impacto)
        riesgos_totales = [p * i for p, i in zip(probabilidades, impactos)]
        
        # Crear figura con múltiples subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Matriz de Riesgo (Probabilidad vs Impacto)
        ax1 = plt.subplot(2, 3, 1)
        colors = {'Técnico': 'blue', 'Datos': 'red', 'Operacional': 'green'}
        for i, cat in enumerate(set(categorias)):
            mask = [c == cat for c in categorias]
            probs_cat = [p for p, m in zip(probabilidades, mask) if m]
            impacts_cat = [i for i, m in zip(impactos, mask) if m]
            names_cat = [n.replace('_', ' ') for n, m in zip(nombres, mask) if m]
            
            scatter = ax1.scatter(probs_cat, impacts_cat, c=colors[cat], 
                                label=cat, s=100, alpha=0.7)
            
            # Añadir etiquetas
            for j, name in enumerate(names_cat):
                ax1.annotate(name[:15] + '...', (probs_cat[j], impacts_cat[j]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax1.set_xlabel('Probabilidad')
        ax1.set_ylabel('Impacto')
        ax1.set_title('Matriz de Riesgo: Probabilidad vs Impacto')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # Añadir líneas de severidad
        ax1.axline((0.3, 0.3), (0.7, 0.7), color='orange', linestyle='--', alpha=0.5, label='Riesgo Medio')
        ax1.axline((0.5, 0.5), (0.9, 0.9), color='red', linestyle='--', alpha=0.5, label='Riesgo Alto')
        
        # 2. Gráfico de barras por severidad
        ax2 = plt.subplot(2, 3, 2)
        severidad_counts = {}
        for sev in severidades:
            severidad_counts[sev] = severidad_counts.get(sev, 0) + 1
        
        colors_sev = {'Baja': 'green', 'Media': 'orange', 'Alta': 'red', 'Crítica': 'darkred'}
        bars = ax2.bar(severidad_counts.keys(), severidad_counts.values(), 
                      color=[colors_sev[k] for k in severidad_counts.keys()])
        ax2.set_title('Distribución por Severidad')
        ax2.set_ylabel('Número de Riesgos')
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 3. Top 5 riesgos por impacto total
        ax3 = plt.subplot(2, 3, 3)
        top_indices = sorted(range(len(riesgos_totales)), key=lambda i: riesgos_totales[i], reverse=True)[:5]
        top_names = [nombres[i].replace('_', ' ') for i in top_indices]
        top_values = [riesgos_totales[i] for i in top_indices]
        
        bars = ax3.barh(top_names, top_values, color='coral')
        ax3.set_title('Top 5 Riesgos por Impacto Total')
        ax3.set_xlabel('Riesgo Total (Prob × Impacto)')
        
        # Añadir valores en las barras
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.2f}', ha='left', va='center')
        
        # 4. Distribución por categoría
        ax4 = plt.subplot(2, 3, 4)
        categoria_counts = {}
        for cat in categorias:
            categoria_counts[cat] = categoria_counts.get(cat, 0) + 1
        
        wedges, texts, autotexts = ax4.pie(categoria_counts.values(), labels=categoria_counts.keys(), 
                                          autopct='%1.1f%%', startangle=90,
                                          colors=[colors[k] for k in categoria_counts.keys()])
        ax4.set_title('Distribución por Categoría')
        
        # 5. Calidad de datos (si disponible)
        ax5 = plt.subplot(2, 3, 5)
        if data_quality_score > 0:
            # Gráfico de gauge para calidad de datos
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            # Colores basados en score
            if data_quality_score >= 80:
                color = 'green'
                status = 'Excelente'
            elif data_quality_score >= 60:
                color = 'orange'
                status = 'Buena'
            elif data_quality_score >= 40:
                color = 'yellow'
                status = 'Regular'
            else:
                color = 'red'
                status = 'Deficiente'
            
            ax5.fill_between(theta, 0, r, alpha=0.3, color='lightgray')
            score_theta = np.pi * (data_quality_score / 100)
            ax5.fill_between(theta[theta <= score_theta], 0, r[theta <= score_theta], 
                           alpha=0.7, color=color)
            
            ax5.set_ylim(0, 1.2)
            ax5.set_xlim(0, np.pi)
            ax5.set_title(f'Calidad de Datos: {data_quality_score:.1f}%\n({status})')
            ax5.set_xticks([0, np.pi/2, np.pi])
            ax5.set_xticklabels(['0%', '50%', '100%'])
            ax5.set_yticks([])
        else:
            ax5.text(0.5, 0.5, 'Sin datos\ncargados', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Calidad de Datos')
        
        # 6. Heatmap de correlación riesgo-categoría
        ax6 = plt.subplot(2, 3, 6)
        
        # Crear matriz de riesgo por categoría
        categorias_unicas = list(set(categorias))
        matriz_riesgo = np.zeros((len(categorias_unicas), 3))  # 3 niveles de severidad principales
        
        severidad_map = {'Baja': 0, 'Media': 1, 'Alta': 2, 'Crítica': 2}
        
        for i, cat in enumerate(categorias_unicas):
            for j, (nombre, riesgo) in enumerate(riesgos.items()):
                if riesgo['categoria'] == cat:
                    sev_idx = severidad_map[riesgo['severidad']]
                    matriz_riesgo[i, sev_idx] += riesgo['probabilidad'] * riesgo['impacto']
        
        im = ax6.imshow(matriz_riesgo, cmap='Reds', aspect='auto')
        ax6.set_xticks(range(3))
        ax6.set_xticklabels(['Baja-Media', 'Alta', 'Crítica'])
        ax6.set_yticks(range(len(categorias_unicas)))
        ax6.set_yticklabels(categorias_unicas)
        ax6.set_title('Intensidad de Riesgo por Categoría')
        
        # Añadir valores en el heatmap
        for i in range(len(categorias_unicas)):
            for j in range(3):
                text = ax6.text(j, i, f'{matriz_riesgo[i, j]:.2f}',
                               ha="center", va="center", color="white" if matriz_riesgo[i, j] > 0.5 else "black")
        
        plt.tight_layout()
        
        # Guardar gráfico
        riesgos_file = OUTPUT_DIR / f"analisis_riesgos_{analysis_type}.png"
        plt.savefig(riesgos_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logging.debug(f"Visualizaciones de riesgos guardadas en {riesgos_file}")
        
        return riesgos_file

    def mostrar_resultados_riesgos(self, riesgos, include_mitigation, detailed_metrics, 
                                  data_quality_score, analysis_type):
        """Mostrar ventana con resultados detallados del análisis de riesgos"""
        results_window = tk.Toplevel(self.root)
        results_window.title("Resultados del Análisis de Riesgos")
        results_window.geometry("800x600")
        results_window.resizable(True, True)
        
        # Frame principal con scrollbar
        main_frame = tk.Frame(results_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Canvas y scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Título
        title_label = tk.Label(scrollable_frame, text="Análisis de Riesgos - Resultados Detallados", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Resumen ejecutivo
        summary_frame = tk.LabelFrame(scrollable_frame, text="Resumen Ejecutivo", 
                                     font=("Arial", 12, "bold"))
        summary_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Calcular estadísticas
        total_riesgos = len(riesgos)
        riesgos_criticos = sum(1 for r in riesgos.values() if r['severidad'] == 'Crítica')
        riesgos_altos = sum(1 for r in riesgos.values() if r['severidad'] == 'Alta')
        riesgo_promedio = sum(r['probabilidad'] * r['impacto'] for r in riesgos.values()) / total_riesgos
        
        summary_text = f"ANÁLISIS COMPLETADO - Tipo: {analysis_type.upper()}\n\n"
        summary_text += f"• Total de riesgos evaluados: {total_riesgos}\n"
        summary_text += f"• Riesgos críticos: {riesgos_criticos}\n"
        summary_text += f"• Riesgos altos: {riesgos_altos}\n"
        summary_text += f"• Riesgo promedio: {riesgo_promedio:.2f}\n"
        
        if data_quality_score > 0:
            summary_text += f"• Calidad de datos: {data_quality_score:.1f}%\n"
        
        summary_label = tk.Label(summary_frame, text=summary_text, font=("Arial", 10), 
                                justify=tk.LEFT, bg="lightyellow")
        summary_label.pack(anchor="w", padx=10, pady=10, fill=tk.X)
        
        # Riesgos detallados
        for riesgo_nombre, riesgo_data in riesgos.items():
            risk_frame = tk.LabelFrame(scrollable_frame, 
                                     text=f"{riesgo_nombre.replace('_', ' ')} - {riesgo_data['severidad']}", 
                                     font=("Arial", 11, "bold"))
            risk_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Información básica
            info_text = f"Categoría: {riesgo_data['categoria']}\n"
            info_text += f"Descripción: {riesgo_data['descripcion']}\n"
            
            if detailed_metrics:
                info_text += f"Probabilidad: {riesgo_data['probabilidad']:.1%}\n"
                info_text += f"Impacto: {riesgo_data['impacto']:.1%}\n"
                info_text += f"Riesgo Total: {riesgo_data['probabilidad'] * riesgo_data['impacto']:.2f}\n"
            
            info_label = tk.Label(risk_frame, text=info_text, font=("Arial", 9), 
                                 justify=tk.LEFT, bg="white")
            info_label.pack(anchor="w", padx=10, pady=5, fill=tk.X)
            
            # Estrategias de mitigación
            if include_mitigation and 'mitigacion' in riesgo_data:
                mitigation_text = "Estrategias de Mitigación:\n"
                for i, estrategia in enumerate(riesgo_data['mitigacion'], 1):
                    mitigation_text += f"  {i}. {estrategia}\n"
                
                mitigation_label = tk.Label(risk_frame, text=mitigation_text, font=("Arial", 9), 
                                          justify=tk.LEFT, bg="lightgreen", fg="darkgreen")
                mitigation_label.pack(anchor="w", padx=10, pady=(0, 5), fill=tk.X)
        
        # Recomendaciones finales
        recommendations_frame = tk.LabelFrame(scrollable_frame, text="Recomendaciones Prioritarias", 
                                            font=("Arial", 12, "bold"))
        recommendations_frame.pack(fill=tk.X, pady=(15, 0))
        
        # Identificar los 3 riesgos más críticos
        riesgos_ordenados = sorted(riesgos.items(), 
                                 key=lambda x: x[1]['probabilidad'] * x[1]['impacto'], 
                                 reverse=True)[:3]
        
        rec_text = "ACCIONES INMEDIATAS RECOMENDADAS:\n\n"
        for i, (nombre, data) in enumerate(riesgos_ordenados, 1):
            rec_text += f"{i}. {nombre.replace('_', ' ')}\n"
            rec_text += f"   → {data['mitigacion'][0] if data['mitigacion'] else 'Revisar estrategias'}\n\n"
        
        rec_label = tk.Label(recommendations_frame, text=rec_text, font=("Arial", 10), 
                           justify=tk.LEFT, bg="lightcoral", fg="darkred")
        rec_label.pack(anchor="w", padx=10, pady=10, fill=tk.X)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Botón cerrar
        close_btn = tk.Button(scrollable_frame, text="Cerrar Análisis", command=results_window.destroy,
                             bg="#FF5722", fg="white", font=("Arial", 12, "bold"))
        close_btn.pack(pady=(20, 0))
        
        # Mensaje de éxito
        messagebox.showinfo("Análisis de Riesgos Completado", 
                           f"Análisis de riesgos {analysis_type} completado exitosamente.\n"
                           f"Se evaluaron {total_riesgos} riesgos.\n"
                           f"Visualizaciones guardadas en la carpeta 'output'.")

    def geocodificar_direcciones(self):
        if self.df.empty or 'departamento' not in self.df.columns or 'provincia' not in self.df.columns or 'pais' not in self.df.columns:
            messagebox.showwarning("Advertencia", "Por favor, asegúrese de que el archivo CSV contenga las columnas 'departamento', 'provincia' y 'pais'.")
            return

        def geocode_with_retry(address):
            try:
                location = geolocator.geocode(address)
                return location
            except (GeocoderTimedOut, GeocoderServiceError):
                sleep(1)
                return geocode_with_retry(address)

        # Crear ventana de progreso
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Geocodificando Direcciones")
        progress_window.geometry("500x200")
        progress_window.resizable(False, False)
        progress_window.grab_set()  # Hacer la ventana modal
        
        # Centrar la ventana
        progress_window.transient(self.root)
        
        # Frame principal
        main_frame = tk.Frame(progress_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        title_label = tk.Label(main_frame, text="Geocodificando Direcciones", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Información del progreso
        total_rows = len(self.df)
        info_label = tk.Label(main_frame, text=f"Procesando {total_rows} direcciones...", 
                             font=("Arial", 10))
        info_label.pack(pady=(0, 10))
        
        # Barra de progreso
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(main_frame, variable=progress_var, 
                                      maximum=100, length=400, mode='determinate')
        progress_bar.pack(pady=(0, 10))
        
        # Etiqueta de estado
        status_label = tk.Label(main_frame, text="Iniciando geocodificación...", 
                               font=("Arial", 9), fg="blue")
        status_label.pack(pady=(0, 10))
        
        # Etiqueta de progreso numérico
        progress_label = tk.Label(main_frame, text="0 / 0 (0%)", 
                                 font=("Arial", 9), fg="gray")
        progress_label.pack()

        latitudes = []
        longitudes = []
        addresses = []
        
        # Actualizar la ventana para mostrarla
        progress_window.update()

        for index, row in self.df.iterrows():
            # Actualizar información de progreso
            current_progress = ((index + 1) / total_rows) * 100
            progress_var.set(current_progress)
            
            address = f"{row['departamento']}, {row['provincia']}, {row['pais']}"
            status_label.config(text=f"Procesando: {address[:50]}...")
            progress_label.config(text=f"{index + 1} / {total_rows} ({current_progress:.1f}%)")
            
            # Actualizar la interfaz
            progress_window.update()
            
            # Geocodificar la dirección
            location = geocode_with_retry(address)
            if location:
                latitudes.append(location.latitude)
                longitudes.append(location.longitude)
                addresses.append(location.address)
                status_label.config(text=f"✓ Encontrada: {location.address[:50]}...", fg="green")
            else:
                latitudes.append(None)
                longitudes.append(None)
                addresses.append(None)
                status_label.config(text=f"✗ No encontrada: {address[:50]}...", fg="red")
            
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
        logging.debug(f"Archivo CSV geocodificado guardado en {geocoded_file}")

        # Mostrar estadísticas finales
        successful_geocodes = sum(1 for lat in latitudes if lat is not None)
        failed_geocodes = total_rows - successful_geocodes
        
        status_label.config(text=f"✓ Completado: {successful_geocodes} exitosas, {failed_geocodes} fallidas", fg="green")
        progress_window.update()
        
        # Esperar un momento antes de cerrar
        sleep(1)
        progress_window.destroy()

        messagebox.showinfo("Geocodificación", 
                           f"Geocodificación completada.\n"
                           f"Direcciones procesadas: {total_rows}\n"
                           f"Geocodificaciones exitosas: {successful_geocodes}\n"
                           f"Geocodificaciones fallidas: {failed_geocodes}\n"
                           f"Archivo guardado en: {geocoded_file}")

    def generar_mapa(self):
        if self.df.empty or 'Latitude' not in self.df.columns or 'Longitude' not in self.df.columns:
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
        logging.debug(f"Mapa geoespacial guardado en {mapa_file}")

        webbrowser.open(mapa_file.resolve().as_uri())

        messagebox.showinfo("Generar Mapa", "Mapa generado exitosamente.")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = DataAnalyzer()
    app.run()
