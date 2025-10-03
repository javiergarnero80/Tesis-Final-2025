# Storytelling de la Presentación – "Análisis con IA 2025"

Propósito: contar, en 10–12 minutos, cómo pasamos de datos crudos a decisiones agrícolas accionables, cubriendo cinco ejes: Desarrollos de scripts, Análisis de Datos, Búsqueda de patrones, Display de resultados y Correcciones de estilos.

1) Hook y problema
- La producción agrícola necesita decisiones rápidas con evidencia. Los registros históricos están dispersos y desbalanceados. La tesis propone una aplicación que integre GUI, estadística e IA para transformar CSVs en insights.

2) Desarrollos de scripts: del prototipo al sistema
- Prototipos monolíticos de exploración: [Analisis con IA 2025.py](Analisis con IA 2025.py), [Tesis.py](Tesis.py), [Tesis2025.py](Tesis2025.py).
- Evolución a arquitectura modular para entrega académica: [Universidad Tesis 2025-10/Analisis con IA 2025.py](Universidad Tesis 2025-10/Analisis con IA 2025.py) + módulos [gui.py](Universidad Tesis 2025-10/gui.py), [analysis.py](Universidad Tesis 2025-10/analysis.py), [data_loader.py](Universidad Tesis 2025-10/data_loader.py), [utils.py](Universidad Tesis 2025-10/utils.py).
- Documentación de apoyo y estructura de tesis: [storytelling_codigo_tesis.md](storytelling_codigo_tesis.md), [README.md](README.md), [Universidad Tesis 2025-10/README_ENTREGA.md](Universidad Tesis 2025-10/README_ENTREGA.md), [tesis_estructura.txt](tesis_estructura.txt), [Universidad Tesis 2025-10/tesis_estructura.txt](Universidad Tesis 2025-10/tesis_estructura.txt).
- Decisiones clave de diseño: normalización de columnas, separación GUI/lógica, manejo de errores y logging, generación automática de directorios de salida.

3) Análisis de Datos: de EDA a señales robustas
- Limpieza y validación: columnas obligatorias, casting numérico, eliminación de NaN en subconjuntos críticos.
- EDA cuantitativa: totales y promedios, coeficiente de variación, detección de outliers por IQR, intervalos de confianza con bootstrap.
- Correlación: matriz profesional y ranking de relaciones para priorizar variables predictoras y detectar trade-offs.
- Series temporales: evolución de superficies, producción y rendimiento por campaña; métricas de eficiencia de cosecha.
- Predictivo: línea base con regresión; comparación controlada de modelos (SVR, Random Forest, regresión) y red neuronal para no linealidades.

4) Búsqueda de patrones: qué aprendimos de los datos
- Patrones lineales fuertes (positivos y negativos) entre superficie y producción; variables independientes útiles para diversificar.
- Heterogeneidad territorial: provincias con predominio de niveles de riesgo (alto/medio/bajo) y especializaciones por cultivo.
- Tendencias por cultivo: estabilidad vs. variabilidad (coeficiente de variación) para priorizar cultivos confiables.
- Señales predictivas: mejoras de R² y reducción de error al pasar de baseline a modelos optimizados.

5) Display de resultados: cómo lo ve el jurado/usuario
- Gráficos académicos en 4 paneles (totales, promedios con IC, CV, min–prom–max) exportados a 300 DPI.
- Heatmap de correlaciones, comparativas de modelos, distribución de errores, series temporales con proyección a 5 años.
- Mapas interactivos HTML (Folium) para distribución y riesgo; aperturas automáticas en navegador.
- Mensajes explicativos y ventanas con scroll en la GUI para interpretación ejecutiva de cada análisis.

6) Correcciones de estilos: rigor visual y consistencia
- Paleta sobria y legible; tema Seaborn whitegrid; fuentes serif para piezas de tesis; anotaciones legibles en barras y líneas.
- Ejes con magnitudes humanas (K/M/B) y etiquetas rotadas; leyendas y guías de interpretación en panel de correlaciones.
- Nombres de archivos seguros y consistentes; guardado automático en carpetas dedicadas; DPI y bounding box para impresión.

Guion sugerido de diapositivas (10–12 min)
- Slide 1: Título + problema real del productor.
- Slide 2: Objetivo y propuesta de valor.
- Slide 3: Desarrollos de scripts (línea de tiempo: prototipo → modular). Referencias: [Analisis con IA 2025.py](Analisis con IA 2025.py), [Universidad Tesis 2025-10/Analisis con IA 2025.py](Universidad Tesis 2025-10/Analisis con IA 2025.py), [gui.py](Universidad Tesis 2025-10/gui.py), [analysis.py](Universidad Tesis 2025-10/analysis.py), [data_loader.py](Universidad Tesis 2025-10/data_loader.py), [utils.py](Universidad Tesis 2025-10/utils.py).
- Slide 4: Pipeline de Análisis de Datos (limpieza → EDA → correlación → temporal → predictivo).
- Slide 5: Búsqueda de patrones (top correlaciones, riesgos por provincia, estabilidad por cultivo).
- Slide 6: Display de resultados (4 paneles + mapas + comparativas de modelos).
- Slide 7: Correcciones de estilos (antes/después) y criterios de legibilidad.
- Slide 8: Impacto y casos de uso (planificación, inversión, mitigación de riesgos).
- Slide 9: Limitaciones y trabajo futuro (datos, clima, web, IoT, deep learning).
- Slide 10: Cierre con llamada a acción (adopción en campañas piloto).

Notas para el orador
- Anclar cada visual con “qué significa” y “qué haría un productor con esto”.
- Evitar jerga técnica sin interpretación; mostrar número + decisión.
- Si R² no es alto, explicar limitaciones de datos y próximos pasos.
- En mapas, destacar 1–2 zonas y una recomendación accionable.

Apoyos y anexos para la defensa
- Guión narrativo extendido: [storytelling_codigo_tesis.md](storytelling_codigo_tesis.md).
- Documentación técnica: [README.md](README.md), [Universidad Tesis 2025-10/README_ENTREGA.md](Universidad Tesis 2025-10/README_ENTREGA.md).
- Marco académico: [tesis_estructura.txt](tesis_estructura.txt), [Universidad Tesis 2025-10/tesis_estructura.txt](Universidad Tesis 2025-10/tesis_estructura.txt).
- Demos en vivo: abrir la app desde [Analisis con IA 2025.py](Analisis con IA 2025.py) o la versión modular.

Resumen final en una frase
Datos agrícolas fragmentados se convierten en decisiones claras mediante una aplicación que integra scripts evolucionados, análisis estadístico y de IA, detección de patrones útiles, visualizaciones académicas y un estilo gráfico consistente apto para tesis.

## Agradecimientos

Queremos expresar nuestro más sincero agradecimiento a nuestras familias, por su apoyo incondicional, paciencia y motivación constante durante este proceso de tesis. A nuestro mentor de tesis, Gustavo García, por su guía experta, consejos valiosos y dedicación que nos ayudaron a navegar los desafíos técnicos y académicos. Y a quienes ya no están con nosotros físicamente, pero que desde donde estén nos apoyaron y nos inspiraron para llegar a esta instancia. Gracias por ser parte de este logro.