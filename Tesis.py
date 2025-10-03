import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import folium
import webbrowser
from pathlib import Path
import logging
from time import sleep
from geopy.geocoders import Nominatim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import argparse
import sys
import os
import matplotlib as mpl
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Asegura directorio de configuración de Matplotlib y backend para CLI/headless
os.environ.setdefault("MPLCONFIGDIR", str((OUTPUT_DIR / ".mpl").resolve()))
(OUTPUT_DIR / ".mpl").mkdir(parents=True, exist_ok=True)

# Detecta si se está ejecutando en modo CLI (antes de importar pyplot)
_IS_CLI_INVOCATION = any(arg in sys.argv for arg in ("--cli", "--analysis", "--csv"))
if _IS_CLI_INVOCATION:
    try:
        mpl.use("Agg")
    except Exception as _e:
        logging.debug(f"No se pudo forzar backend Agg: {_e}")

import matplotlib.pyplot as plt
import seaborn as sns

geolocator = Nominatim(user_agent="geopy/1.22.0 (github.com/geopy/geopy)")

class DataAnalyzer:
    def __init__(self, headless: bool = False, open_browser: bool = True):
        self.headless = headless
        self.open_browser = open_browser
        self.df = pd.DataFrame()
        self.mapa_generado = False

        # GUI setup only when not headless
        if not self.headless:
            self.root = tk.Tk()
            self.root.title("Aplicación de Análisis de Datos")

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
        else:
            self.root = None

    # Notificación unificada (GUI vs CLI)
    def _info(self, title: str, message: str):
        if self.headless:
            logging.info(f"{title}: {message}")
        else:
            messagebox.showinfo(title, message)

    def _warning(self, title: str, message: str):
        if self.headless:
            logging.warning(f"{title}: {message}")
        else:
            messagebox.showwarning(title, message)

    def _error(self, title: str, message: str):
        if self.headless:
            logging.error(f"{title}: {message}")
        else:
            messagebox.showerror(title, message)

    def cargar_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                logging.debug(f"Archivo CSV cargado: {file_path}")
                self._info("Cargar CSV", "Archivo CSV cargado exitosamente.")
            except Exception as e:
                logging.error(f"Error al cargar el archivo CSV: {e}")
                self._error("Error", f"Ocurrió un error al cargar el archivo CSV: {e}")

    def produccion_total_por_provincia(self):
        if self.df.empty or 'provincia' not in self.df.columns or 'produccion' not in self.df.columns:
            self._warning("Advertencia", "Por favor, asegúrese de que el archivo CSV contenga las columnas 'provincia' y 'produccion'.")
            return

        produccion_por_provincia = self.df.groupby('provincia')['produccion'].sum().sort_values(ascending=False)

        produccion_por_provincia.plot(kind='bar')
        plt.title("Producción Total por Provincia")
        plt.ylabel("Producción [Tn]")
        plt.xlabel("Provincias")
        plt.tight_layout()

        produccion_por_provincia_file = OUTPUT_DIR / "produccion_por_provincia.png"
        plt.savefig(produccion_por_provincia_file)
        if not self.headless:
            plt.show()
        logging.debug(f"Gráfica de producción total por provincia guardada en {produccion_por_provincia_file}")

        self._info("Producción Total por Provincia", f"Gráfica de producción total por provincia guardada en {produccion_por_provincia_file}")

    def correlacion_superficie_produccion(self, provincia: str = None):
        if self.df.empty:
            self._warning("Advertencia", "El DataFrame está vacío. Por favor, cargue un archivo CSV primero.")
            return

        if 'provincia' not in self.df.columns or 'sup_sembrada' not in self.df.columns or 'sup_cosechada' not in self.df.columns:
            self._warning("Advertencia", "Por favor, asegúrese de que el archivo CSV contenga las columnas 'provincia', 'sup_sembrada' y 'sup_cosechada'.")
            return

        if provincia is None:
            if self.headless:
                self._warning("Advertencia", "En modo CLI debe indicar --provincia para 'correlacion_sup'.")
                return
            provincias = self.df['provincia'].unique()
            selected_provincia = self.ask_option("Seleccionar Provincia", "Seleccione la provincia:", provincias)
        else:
            selected_provincia = provincia

        if not selected_provincia:
            return

        df_provincia = self.df[self.df['provincia'] == selected_provincia]

        if df_provincia.empty:
            self._warning("Advertencia", "No se encontraron datos para la provincia seleccionada.")
            return

        df_provincia[['sup_sembrada', 'sup_cosechada']] = df_provincia[['sup_sembrada', 'sup_cosechada']].apply(pd.to_numeric, errors='coerce')
        df_provincia = df_provincia.dropna(subset=['sup_sembrada', 'sup_cosechada'])

        if df_provincia.empty:
            self._warning("Advertencia", "Después de la conversión a numérico, no se encontraron datos válidos para calcular la correlación.")
            return

        try:
            correlacion = df_provincia[['sup_sembrada', 'sup_cosechada']].corr().iloc[0, 1]
            if correlacion >= 0.7:
                self._info("Correlación Sup. Sembrada-Sup. Cosechada", f"Correlación alta ({correlacion:.2f}). Sugerencia: Explorar variedades de cultivos que optimicen la superficie cosechada.")
            elif correlacion <= 0.3:
                self._info("Correlación Sup. Sembrada-Sup. Cosechada", f"Correlación baja ({correlacion:.2f}). Sugerencia: Revisar prácticas de cultivo y factores ambientales.")
            else:
                self._info("Correlación Sup. Sembrada-Sup. Cosechada", f"Correlación moderada ({correlacion:.2f}). Considerar diversificación de cultivos.")
        except Exception as e:
            logging.error(f"Error al calcular la correlación: {e}")
            self._error("Error", f"Ocurrió un error al calcular la correlación: {e}")

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
        combobox.current(0)

        button = tk.Button(dialog, text="Aceptar", command=dialog.destroy)
        button.pack(pady=10)

        dialog.grab_set()
        dialog.wait_window()

        selected_option = combobox_value.get()
        return selected_option

    def sumar_columnas(self):
        if self.df.empty:
            self._warning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return

        suma_columnas = self.df.select_dtypes(include=[float, int]).sum()
        suma_columnas.plot(kind='bar')
        plt.title("Suma de Columnas Numéricas")
        plt.ylabel("Suma")
        plt.xlabel("Columnas")
        plt.tight_layout()

        suma_columnas_file = OUTPUT_DIR / "suma_columnas.png"
        plt.savefig(suma_columnas_file)
        if not self.headless:
            plt.show()
        logging.debug(f"Gráfica de suma de columnas guardada en {suma_columnas_file}")

        self._info("Suma de Columnas", f"Gráfica de suma de columnas guardada en {suma_columnas_file}")

    def analisis_temporal(self):
        if self.df.empty or 'campaña' not in self.df.columns:
            self._warning("Advertencia", "Por favor, asegúrese de que el archivo CSV contenga la columna 'campaña'.")
            return

        if 'produccion' not in self.df.columns:
            self._warning("Advertencia", "Por favor, asegúrese de que el archivo CSV contenga la columna 'produccion'.")
            return

        self.df['campaña'] = pd.to_datetime(self.df['campaña'], errors='coerce')
        df_temporal = self.df.groupby(self.df['campaña'].dt.year).sum(numeric_only=True)

        df_temporal.plot(y='produccion', kind='line')
        plt.title("Análisis Temporal de Producción")
        plt.ylabel("Producción [Tn]")
        plt.xlabel("Año")
        plt.tight_layout()

        temporal_file = OUTPUT_DIR / "analisis_temporal.png"
        plt.savefig(temporal_file)
        if not self.headless:
            plt.show()
        logging.debug(f"Gráfica de análisis temporal guardada en {temporal_file}")

        self._info("Análisis Temporal", f"Gráfica de análisis temporal guardada en {temporal_file}")

    def analisis_geoespacial(self):
        if self.df.empty or 'pais' not in self.df.columns or 'provincia' not in self.df.columns:
            self._warning("Advertencia", "Por favor, asegúrese de que el archivo CSV contenga las columnas 'pais' y 'provincia'.")
            return

        self.df['coordenadas'] = self.df.apply(lambda row: self.geocodificar(row['provincia'] + ', ' + row['pais']), axis=1)
        self.df = self.df.dropna(subset=['coordenadas'])

        if self.df.empty:
            self._warning("Advertencia", "No se pudieron obtener coordenadas para las ubicaciones.")
            return

        mapa = folium.Map(location=[self.df['coordenadas'].apply(lambda x: x[0]).mean(), self.df['coordenadas'].apply(lambda x: x[1]).mean()], zoom_start=6)

        for _, row in self.df.iterrows():
            folium.Marker([row['coordenadas'][0], row['coordenadas'][1]], popup=row['provincia']).add_to(mapa)

        mapa_file = OUTPUT_DIR / "mapa_cultivos.html"
        mapa.save(mapa_file)
        if (not self.headless) or self.open_browser:
            webbrowser.open(str(mapa_file))  # Convertir a cadena
        logging.debug(f"Mapa geoespacial guardado en {mapa_file}")

        self._info("Análisis Geoespacial", f"Mapa geoespacial guardado en {mapa_file}")

    def geocodificar(self, direccion):
        try:
            location = geolocator.geocode(direccion)
            if location:
                return (location.latitude, location.longitude)
            else:
                return None
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logging.error(f"Error en la geocodificación: {e}")
            return None

    def geocodificar_direcciones(self):
        if self.df.empty or 'provincia' not in self.df.columns or 'pais' not in self.df.columns:
            self._warning("Advertencia", "Por favor, asegúrese de que el archivo CSV contenga las columnas 'pais' y 'provincia'.")
            return

        self.df['coordenadas'] = self.df.apply(lambda row: self.geocodificar(row['provincia'] + ', ' + row['pais']), axis=1)
        self.df = self.df.dropna(subset=['coordenadas'])

        if self.df.empty:
            self._warning("Advertencia", "No se pudieron obtener coordenadas para las ubicaciones.")
            return

        self._info("Geocodificación", "Geocodificación completada exitosamente.")

    def generar_mapa(self):
        if not self.mapa_generado:
            self.analisis_geoespacial()
            self.mapa_generado = True
        else:
            if (not self.headless) or self.open_browser:
                webbrowser.open(str(OUTPUT_DIR / "mapa_cultivos.html"))  # Convertir a cadena

    def analisis_correlacion(self):
        if self.df.empty:
            self._warning("Advertencia", "El DataFrame está vacío. Por favor, cargue un archivo CSV primero.")
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
        if not self.headless:
            plt.show()
        logging.debug(f"Matriz de correlación guardada en {correlacion_file}")

        self._info("Análisis de Correlación", f"Matriz de correlación guardada en {correlacion_file}")

    def modelos_predictivos(self):
        if self.df.empty or 'sup_sembrada' not in self.df.columns or 'produccion' not in self.df.columns:
            self._warning("Advertencia", "El DataFrame debe contener 'sup_sembrada' y 'produccion'.")
            return

        X = self.df[['sup_sembrada']].values
        y = self.df['produccion'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self._info("Modelo Predictivo", f"Error cuadrático medio (MSE): {mse:.2f}\nCoeficiente de determinación (R2): {r2:.2f}")

    def analisis_riesgos(self):
        if self.df.empty or 'produccion' not in self.df.columns:
            self._warning("Advertencia", "El DataFrame debe contener la columna 'produccion'.")
            return

        # Normalización de datos
        scaler = StandardScaler()
        df_prod = self.df[['produccion']].dropna()
        if df_prod.empty:
            self._warning("Advertencia", "No hay datos válidos en 'produccion' para analizar.")
            return
        df_normalizado = scaler.fit_transform(df_prod)

        # Reducción de dimensionalidad
        if df_normalizado.shape[1] >= 2:
            pca = PCA(n_components=2)
            df_reducido = pca.fit_transform(df_normalizado)
        else:
            # Si solo hay 1 feature, crear una segunda dimensión cero para graficar
            df_reducido = np.hstack([df_normalizado, np.zeros_like(df_normalizado)])

        # Clustering
        n_samples = df_reducido.shape[0]
        n_clusters = min(3, n_samples) if n_samples > 0 else 0
        if n_clusters < 1:
            self._warning("Advertencia", "No hay suficientes muestras para clustering.")
            return
        if n_clusters == 1:
            clusters = np.zeros(n_samples, dtype=int)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(df_reducido)

        self.df['cluster'] = clusters
        plt.figure(figsize=(10, 8))
        plt.scatter(df_reducido[:, 0], df_reducido[:, 1], c=clusters, cmap='viridis')
        plt.title("Clustering de Producción")
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.colorbar(label='Cluster')

        clustering_file = OUTPUT_DIR / "clustering_produccion.png"
        plt.savefig(clustering_file)
        if not self.headless:
            plt.show()
        logging.debug(f"Gráfica de clustering guardada en {clustering_file}")

        self._info("Análisis de Riesgos", f"Gráfica de clustering de producción guardada en {clustering_file}")

def _run_cli():
    parser = argparse.ArgumentParser(description="Análisis de datos (GUI o CLI)")
    parser.add_argument("--cli", action="store_true", help="Ejecutar en modo CLI (sin GUI)")
    parser.add_argument("--csv", help="Ruta al archivo CSV")
    parser.add_argument(
        "--analysis",
        choices=[
            "sumar",
            "temporal",
            "geoespacial",
            "correlacion",
            "modelos",
            "riesgos",
            "correlacion_sup",
            "produccion_provincia",
            "geocode",
            "mapa",
        ],
        help="Análisis a ejecutar",
    )
    parser.add_argument("--provincia", help="Provincia (para correlacion_sup)")
    parser.add_argument("--output-dir", help="Directorio de salida")
    parser.add_argument("--open-browser", action="store_true", help="Abrir mapas generados en el navegador")

    args = parser.parse_args()

    # Si no se solicita CLI explícitamente, pero no hay análisis/CSV, caemos en GUI
    if not (args.cli or args.csv or args.analysis):
        return None

    # Validaciones CLI
    if not args.csv:
        print("Error: Debe indicar --csv en modo CLI", file=sys.stderr)
        sys.exit(2)

    if args.output_dir:
        global OUTPUT_DIR
        OUTPUT_DIR = Path(args.output_dir)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    analyzer = DataAnalyzer(headless=True, open_browser=bool(args.open_browser))

    try:
        analyzer.df = pd.read_csv(args.csv)
        logging.debug(f"CSV cargado (CLI): {args.csv}")
    except Exception as e:
        print(f"Error al leer CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.analysis:
        print("Modo CLI activo, pero no se indicó --analysis. Nada para ejecutar.")
        return 0

    # Despacho de análisis
    a = args.analysis
    if a == "sumar":
        analyzer.sumar_columnas()
    elif a == "temporal":
        analyzer.analisis_temporal()
    elif a == "geoespacial":
        analyzer.analisis_geoespacial()
    elif a == "correlacion":
        analyzer.analisis_correlacion()
    elif a == "modelos":
        analyzer.modelos_predictivos()
    elif a == "riesgos":
        analyzer.analisis_riesgos()
    elif a == "correlacion_sup":
        analyzer.correlacion_superficie_produccion(provincia=args.provincia)
    elif a == "produccion_provincia":
        analyzer.produccion_total_por_provincia()
    elif a == "geocode":
        analyzer.geocodificar_direcciones()
    elif a == "mapa":
        analyzer.generar_mapa()

    return 0


if __name__ == "__main__":
    # Si se pidió CLI, ejecútalo y sal; de lo contrario, lanza GUI
    result = _run_cli()
    if result is None:
        app = DataAnalyzer()
        app.root.mainloop()
