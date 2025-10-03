import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Cargar el archivo CSV
file_path = '/mnt/data/estimaciones_agricolas_2023_10_cleaned.csv'
df = pd.read_csv(file_path)

# Mostrar las primeras filas del datasetprint("Primeras filas del dataset:")
print(df.head())

# Análisis Exploratorio de Datos (EDA)print("\nInformación general del dataset:")
print(df.info())

print("\nEstadísticas descriptivas:")
print(df.describe())

# Geocodificación de las ubicaciones geográficas (provincia y departamento)
geolocator = Nominatim(user_agent="geoapiExercises")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# Aquí estaba el error, debe haber un espacio entre 'def' y 'geocode_location'defgeocode_location(row):
    location = f"{row['departamento']}, {row['provincia']}, Argentina"try:
        location = geolocator.geocode(location)
        return location.latitude, location.longitude
    except:
        returnNone, Noneprint("\nGeocodificando ubicaciones geográficas...")
df[['latitude', 'longitude']] = df.apply(geocode_location, axis=1, result_type='expand')

# Filtrar filas con geocodificación exitosa
df_geo = df.dropna(subset=['latitude', 'longitude'])

# Crear un GeoDataFrame
gdf = gpd.GeoDataFrame(df_geo, geometry=gpd.points_from_xy(df_geo.longitude, df_geo.latitude))

# Visualizar la distribución geográfica de los cultivos
plt.figure(figsize=(10, 10))
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world[world.name == "Argentina"].plot(color='white', edgecolor='black')

gdf.plot(ax=ax, marker='o', color='red', markersize=5)
plt.title("Distribución geográfica de los cultivos en Argentina")
plt.show()

# Análisis Temporal: Evolución de los cultivos por campaña
df['campaña'] = pd.to_datetime(df['campaña'], format='%Y-%Y')
df_grouped = df.groupby(['campaña', 'cultivo']).agg({'sup_sembrada': 'sum', 'produccion': 'sum'}).reset_index()

# Gráfico de la evolución de la superficie sembrada por campaña y cultivo
plt.figure(figsize=(12, 6))
for cultivo in df_grouped['cultivo'].unique():
    cultivo_df = df_grouped[df_grouped['cultivo'] == cultivo]
    plt.plot(cultivo_df['campaña'], cultivo_df['sup_sembrada'], marker='o', label=cultivo)

plt.title("Evolución de la superficie sembrada por cultivo")
plt.xlabel("Campaña")
plt.ylabel("Superficie Sembrada (ha)")
plt.legend()
plt.grid(True)
plt.show()

# Guardar el dataset geocodificado en un nuevo CSV
output_file = '/mnt/data/estimaciones_agricolas_geocodificado.csv'
df_geo.to_csv(output_file, index=False)
print(f"Dataset geocodificado guardado en: {output_file}")
