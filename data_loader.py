"""
Data Loader Module

This module handles CSV file loading and validation for the agricultural analysis application.
"""

import pandas as pd
import logging
from tkinter import filedialog, messagebox


class FileHandler:
    """
    Class for handling CSV file loading and validation.

    Provides methods to load CSV files, validate their structure,
    and perform basic data cleaning operations.
    """

    @staticmethod
    def load_csv():
        """
        Load a CSV file selected by the user.

        Returns:
            pd.DataFrame: Loaded DataFrame if successful, empty DataFrame otherwise.

        Raises:
            Displays error messages to user via messagebox on failure.
        """
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()  # Clean whitespace from column names
                logging.info(f"CSV file loaded successfully: {file_path}")
                messagebox.showinfo("Load CSV", "CSV file loaded successfully.")
                return df
            except pd.errors.EmptyDataError:
                logging.error("CSV file is empty.")
                messagebox.showerror("Error", "CSV file is empty.")
            except pd.errors.ParserError:
                logging.error("CSV parsing error.")
                messagebox.showerror("Error", "CSV parsing error.")
            except Exception as e:
                logging.error(f"Error loading CSV file: {e}")
                messagebox.showerror("Error", f"Error loading CSV file: {e}")
        return pd.DataFrame()

    @staticmethod
    def validate_columns(df, required_columns):
        """
        Validate that required columns exist in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to validate.
            required_columns (list): List of required column names.

        Returns:
            bool: True if all required columns exist, False otherwise.
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            messagebox.showerror("Validation Error",
                               f"Missing required columns: {', '.join(missing_columns)}")
            return False
        return True