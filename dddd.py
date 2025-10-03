import pandas as pd
import re
import os
import tkinter as tk
from tkinter import filedialog, messagebox

class CSVFormatter:
    """Clase para formatear archivos de entrada y convertirlos a un formato CSV listo para la aplicación."""

    REQUIRED_COLUMNS = ['provincia', 'campaña', 'sup_sembrada', 'sup_cosechada', 'produccion', 'rendimiento', 'departamento', 'pais']

    @staticmethod
    def format_input_file(input_file, output_file):
        """
        Formatea el archivo de entrada, asegurándose de que tenga las columnas necesarias y lo guarda como un CSV listo.
        :param input_file: Ruta al archivo de entrada (puede ser CSV, Excel, etc.)
        :param output_file: Ruta al archivo CSV formateado que será generado.
        """
        # Verificar si el archivo de entrada existe
        if not os.path.exists(input_file):
            messagebox.showerror("Error", f"El archivo {input_file} no existe.")
            return

        # Detectar el tipo de archivo y cargarlo con manejo de codificaciones
        try:
            if input_file.endswith('.csv'):
                try:
                    df = pd.read_csv(input_file, sep=';', encoding='latin-1', on_bad_lines='skip', engine='python')
                except UnicodeDecodeError:
                    df = pd.read_csv(input_file, sep=';', encoding='ISO-8859-1', on_bad_lines='skip', engine='python')
            elif input_file.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(input_file)
            else:
                raise ValueError("Formato de archivo no soportado. Usa CSV o Excel.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al leer el archivo: {str(e)}")
            return

        # Mostrar las columnas originales detectadas antes de renombrar
        print("Columnas originales detectadas:", df.columns.tolist())

        # Limpiar y renombrar columnas
        df.columns = df.columns.str.strip()  # Eliminar espacios en blanco de los nombres de columnas

        # Reemplazar las columnas corruptas con las que necesitamos
        df.columns = df.columns.str.replace(r'Campa[^\w\s]*a', 'campaña', regex=True)
        df.columns = df.columns.str.replace(r'Producci[^\w\s]*n \(Tn\)', 'produccion', regex=True)
        df.columns = df.columns.str.replace(r'Sup\. Sembrada \(Ha\)', 'sup_sembrada', regex=True)
        df.columns = df.columns.str.replace(r'Sup\. Cosechada \(Ha\)', 'sup_cosechada', regex=True)
        df.columns = df.columns.str.replace(r'Rendimiento \(Kg/Ha\)', 'rendimiento', regex=True)

        # Mostrar las columnas después del renombrado
        print("Columnas después del renombrado:", df.columns.tolist())

        # Mapeo manual de las columnas que necesitamos
        column_mapping = {
            'Provincia': 'provincia',
            'Departamento': 'departamento',
            'Cultivo': 'cultivo',
            'campana': 'campaña',  # ya corregido antes
            'sup_sembrada': 'sup_sembrada',
            'sup_cosechada': 'sup_cosechada',
            'produccion': 'produccion',
            'rendimiento': 'rendimiento'
        }

        # Aplicar el renombrado de columnas
        df.rename(columns=column_mapping, inplace=True)

        # Añadir la columna 'Pais' si no existe y llenarla con un valor por defecto
        if 'pais' not in df.columns:
            df['pais'] = 'Argentina'

        # Validar que todas las columnas requeridas estén presentes
        missing_columns = [col for col in CSVFormatter.REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            messagebox.showerror("Error", f"Faltan las siguientes columnas requeridas: {', '.join(missing_columns)}")
            return

        # Rellenar valores faltantes (si es necesario)
        df.fillna('', inplace=True)

        # Limpiar datos no numéricos en columnas que deberían ser numéricas
        for column in ['sup_sembrada', 'sup_cosechada', 'produccion', 'rendimiento']:
            df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0)

        # Guardar el archivo CSV formateado
        try:
            df.to_csv(output_file, index=False)
            messagebox.showinfo("Éxito", f"Archivo formateado guardado en: {output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar el archivo: {str(e)}")

    @staticmethod
    def clean_column_name(column_name):
        """
        Limpia el nombre de las columnas eliminando caracteres especiales y poniendo en formato correcto.
        :param column_name: Nombre original de la columna
        :return: Nombre de columna limpio
        """
        column_name = re.sub(r'[^\w\s]', '', column_name)  # Eliminar caracteres especiales
        return column_name.strip().lower().replace(' ', '_')  # Convertir a minúsculas y usar _ en lugar de espacios


def seleccionar_archivos():
    """Función para seleccionar archivos de entrada y salida."""
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter

    # Seleccionar el archivo de entrada
    input_file = filedialog.askopenfilename(title="Seleccione el archivo de entrada", 
                                            filetypes=[("Archivos CSV o Excel", "*.csv *.xls *.xlsx")])
    if not input_file:
        messagebox.showwarning("Advertencia", "Debe seleccionar un archivo de entrada.")
        return

    # Seleccionar la ubicación de guardado del archivo de salida
    output_file = filedialog.asksaveasfilename(title="Guardar archivo como", 
                                               defaultextension=".csv",
                                               filetypes=[("Archivo CSV", "*.csv")])
    if not output_file:
        messagebox.showwarning("Advertencia", "Debe seleccionar una ubicación de salida.")
        return

    # Formatear el archivo
    CSVFormatter.format_input_file(input_file, output_file)


if __name__ == "__main__":
    seleccionar_archivos()
