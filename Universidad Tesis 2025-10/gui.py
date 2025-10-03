"""
GUI Module

This module contains the graphical user interface for the agricultural analysis application.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import re
from data_loader import FileHandler
from analysis import StatisticalAnalyzer, PredictiveModeler, GeospatialAnalyzer
from utils import DataPreprocessing
import logging


class DataAnalyzerApp:
    """
    Main application class for the agricultural data analyzer GUI.
    """

    def __init__(self):
        """
        Initialize the application with GUI setup.
        """
        self.root = tk.Tk()
        self.root.title("Agricultural Data Analysis Application")
        self.root.geometry("600x400")
        self.df = pd.DataFrame()
        self.setup_menu()

    def _check_csv_loaded(self):
        """
        Check if CSV is loaded and show warning if not.

        Returns:
            bool: True if CSV is loaded, False otherwise.
        """
        if self.df.empty:
            messagebox.showwarning("Warning", "Please load a CSV file first.")
            return False
        return True

    def setup_menu(self):
        """
        Set up the application menu.
        """
        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)

        # File menu
        self.file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Load CSV", command=self.load_csv)
        self.file_menu.add_command(label="Exit", command=self.root.quit)

        # Analysis menu
        self.analysis_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Analysis", menu=self.analysis_menu)
        self.analysis_menu.add_command(label="Sum Columns", command=self.sum_columns)
        self.analysis_menu.add_command(label="Temporal Analysis", command=self.temporal_analysis)
        self.analysis_menu.add_command(label="Correlation Analysis", command=self.correlation_analysis)
        self.analysis_menu.add_command(label="Predictive Models", command=self.predictive_models)
        self.analysis_menu.add_command(label="Crop Classification", command=self.crop_classification)
        self.analysis_menu.add_command(label="Risk Analysis", command=self.risk_analysis)
        self.analysis_menu.add_command(label="Correlation Sup. Sembrada-Sup. Cosechada",
                                     command=self.correlation_sup_sembrada_cosechada)
        self.analysis_menu.add_command(label="Production Total by Province",
                                     command=self.production_total_by_province)
        self.analysis_menu.add_command(label="Crop Evolution by Campaign",
                                     command=self.evolution_cultivos_por_campaña)
        self.analysis_menu.add_command(label="Production Trends by Crop",
                                     command=self.tendencias_produccion_por_cultivo)
        self.analysis_menu.add_command(label="AI Trend Prediction",
                                     command=self.prediccion_tendencias_ia)
        self.analysis_menu.add_command(label="Neural Network Analysis",
                                     command=self.analisis_predictivo_nn)
        self.analysis_menu.add_command(label="Top Crops Production",
                                     command=self.produccion_top_cultivos)

        # Geocoding menu
        self.geocodificacion_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Geocoding", menu=self.geocodificacion_menu)
        self.geocodificacion_menu.add_command(label="Geocode Addresses",
                                            command=self.geocodificar_direcciones)
        self.geocodificacion_menu.add_command(label="Generate Map", command=self.generar_mapa)

    def load_csv(self):
        """
        Load CSV file using FileHandler.
        """
        self.df = FileHandler.load_csv()

    # Analysis methods (wrappers for analyzer classes)
    def sum_columns(self):
        if not self._check_csv_loaded():
            return
        StatisticalAnalyzer.sum_columns(self.df)

    def temporal_analysis(self):
        if not self._check_csv_loaded():
            return
        StatisticalAnalyzer.temporal_analysis(self.df)

    def correlation_analysis(self):
        if not self._check_csv_loaded():
            return
        StatisticalAnalyzer.correlation_analysis(self.df)

    def predictive_models(self):
        if not self._check_csv_loaded():
            return
        PredictiveModeler.linear_regression_model(self.df)

    def crop_classification(self):
        # Implementation needed
        pass

    def risk_analysis(self):
        # Implementation needed
        pass

    def correlation_sup_sembrada_cosechada(self):
        # Implementation needed
        pass

    def production_total_by_province(self):
        # Implementation needed
        pass

    def evolution_cultivos_por_campaña(self):
        # Implementation needed
        pass

    def tendencias_produccion_por_cultivo(self):
        # Implementation needed
        pass

    def prediccion_tendencias_ia(self):
        # Implementation needed
        pass

    def analisis_predictivo_nn(self):
        if not self._check_csv_loaded():
            return
        PredictiveModeler.neural_network_model(self.df)

    def produccion_top_cultivos(self):
        # Implementation needed
        pass

    def geocodificar_direcciones(self):
        if not self._check_csv_loaded():
            return
        GeospatialAnalyzer.geocode_addresses(self.df)

    def generar_mapa(self):
        if not self._check_csv_loaded():
            return
        GeospatialAnalyzer.generate_map(self.df)

    def ask_option(self, title, message, options):
        """
        Show a dialog to select an option.

        Args:
            title (str): Dialog title.
            message (str): Dialog message.
            options (list): List of options.

        Returns:
            str: Selected option or None.
        """
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

        button = ttk.Button(dialog, text="OK", command=dialog.destroy)
        button.pack(pady=10)

        dialog.grab_set()
        dialog.wait_window()

        selected_option = combobox_value.get()
        return selected_option

    @staticmethod
    def safe_file_name(name):
        """
        Return a safe filename for filesystem use.

        Args:
            name (str): Input name.

        Returns:
            str: Safe filename.
        """
        return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')


if __name__ == "__main__":
    app = DataAnalyzerApp()
    app.root.mainloop()