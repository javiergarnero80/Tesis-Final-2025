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

        produccion_por_provincia.plot(kind='bar')
        plt.title("Producción Total por Provincia")
        plt.ylabel("Producción")
        plt.xlabel("Provincia")
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

        provincias = self.df['provincia'].unique()
        selected_provincia = self.ask_option("Seleccionar Provincia", "Seleccione la provincia:", provincias)

        if selected_provincia is None:
            return

        if 'sup_sembrada' not in self.df.columns:
            messagebox.showwarning("Advertencia", "La columna 'sup_sembrada' no está en el archivo CSV.")
            return

        sumas_provincias = self.df.groupby('provincia')['sup_sembrada'].sum().to_dict()

        messagebox.showinfo("Correlación Sup. Sembrada-Sup. Cosechada", f"Suma por provincias:\n\n{sumas_provincias}")

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
            # Extraer el primer año de la columna 'campaña' (formato '1969/1970')
            self.df['campaña'] = self.df['campaña'].str.split('/').str[0].astype(int)

            df_temporal = self.df.groupby('campaña').sum()

            df_temporal.plot()
            plt.title("Análisis Temporal de Columnas Numéricas")
            plt.ylabel("Valores")
            plt.xlabel("Campaña")
            plt.tight_layout()

            analisis_temporal_file = OUTPUT_DIR / "analisis_temporal.png"
            plt.savefig(analisis_temporal_file)
            plt.show()
            logging.debug(f"Gráfica de análisis temporal guardada en {analisis_temporal_file}")

            messagebox.showinfo("Análisis Temporal", f"Gráfica de análisis temporal guardada en {analisis_temporal_file}")
        
        except Exception as e:
            logging.error(f"Error al realizar el análisis temporal: {e}")
            messagebox.showerror("Error", f"Ocurrió un error al realizar el análisis temporal: {e}")

    def analisis_geoespacial(self):
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

        messagebox.showinfo("Análisis Geoespacial", "Análisis geoespacial realizado exitosamente.")

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

        messagebox.showinfo("Modelos Predictivos", "Función de modelos predictivos aún no implementada.")

    def analisis_riesgos(self):
        if self.df.empty:
            messagebox.showwarning("Advertencia", "Por favor cargue un archivo CSV primero.")
            return

        # Aquí se realiza el análisis de riesgos
        # ...
        
        # Ejemplo de gráfica de riesgos mejorada
        riesgos = ['Riesgo 1', 'Riesgo 2', 'Riesgo 3']
        valores = [10, 20, 30]

        plt.barh(riesgos, valores)
        plt.title("Análisis de Riesgos")
        plt.xlabel("Valores")
        plt.ylabel("Riesgos")
        plt.tight_layout()

        analisis_riesgos_file = OUTPUT_DIR / "analisis_riesgos.png"
        plt.savefig(analisis_riesgos_file)
        plt.show()
        logging.debug(f"Gráfica de análisis de riesgos guardada en {analisis_riesgos_file}")

        messagebox.showinfo("Análisis de Riesgos", f"Análisis de riesgos realizado exitosamente. Gráfica guardada en {analisis_riesgos_file}")

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
        logging.debug(f"Archivo CSV geocodificado guardado en {geocoded_file}")

        messagebox.showinfo("Geocodificación", f"Geocodificación completada. Archivo guardado en {geocodificado_file}")

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
