"""
Utilities Module

This module contains utility classes for data preprocessing and visualization.
"""

import unicodedata
import re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class DataPreprocessing:
    """
    Class for data normalization and preprocessing operations.
    """

    @staticmethod
    def normalize_text(text):
        """
        Normalize text by removing special characters and accents.

        Args:
            text (str): Input text to normalize.

        Returns:
            str: Normalized text in lowercase without special characters.
        """
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        text = text.lower().strip()  # Convert to lowercase and strip whitespace
        return text

    @staticmethod
    def denormalize_text(normalized_text, original_texts):
        """
        Denormalize text by finding its original version in a list.

        Args:
            normalized_text (str): Normalized text to find.
            original_texts (list): List of original texts to search in.

        Returns:
            str or None: Original text if found, None otherwise.
        """
        for text in original_texts:
            if DataPreprocessing.normalize_text(text) == normalized_text:
                return text
        return None


class Visualization:
    """
    Class for data visualization operations.
    """

    @staticmethod
    def plot_bar_chart(data, title, xlabel, ylabel, output_file, function_name=""):
        """
        Generate a bar chart and save it to file.

        Args:
            data: Data to plot.
            title (str): Chart title.
            xlabel (str): X-axis label.
            ylabel (str): Y-axis label.
            output_file (str): Path to save the chart.
            function_name (str): Optional function name for suptitle.
        """
        fig = plt.figure(figsize=(12, 8))
        if function_name:
            fig.suptitle(f"{function_name}", fontsize=10, y=0.98, ha='left', x=0.02,
                        style='italic', alpha=0.7)
        data.plot(kind='bar', color='skyblue')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()
        logging.info(f"Chart saved to {output_file}")