# Funciones adicionales para completar el análisis geoespacial

def analisis_geoespacial_completo_continuacion(self, num_clusters, analysis_var, include_stats, generate_heatmap, export_results):
    """Análisis geoespacial completo combinando todos los métodos - CONTINUACIÓN"""
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
