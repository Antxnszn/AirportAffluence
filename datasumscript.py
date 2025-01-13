import pandas as pd
import os

# Ruta del archivo CSV
file_path = r'FilteredAirportsData.csv'

# Cargar los datos
data = pd.read_csv(file_path)

# Asegúrate de que las columnas de fechas y aeropuertos sean correctas
data['data_dte'] = pd.to_datetime(data['data_dte'], errors='coerce')
data = data[data['data_dte'].notna()]  # Eliminar filas con fechas inválidas

# Extraer año y mes de la columna de fechas
data['Year'] = data['data_dte'].dt.year
data['Month'] = data['data_dte'].dt.month

# Agrupar por Año, Mes y Aeropuerto, y sumar la columna "Total"
resultados = data.groupby(['Year', 'Month', 'usg_apt'])['Total'].sum().reset_index()

# Ruta de salida para el archivo
output_path = r'C:\Users\rnavarro\Documents\Suma_Total_por_Mes_Año_Aeropuerto.csv'

# Crear el directorio si no existe
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir) and output_dir != '':
    os.makedirs(output_dir)

# Guardar los resultados en un archivo CSV
resultados.to_csv(output_path, index=False)
print(f"Archivo guardado en: {output_path}")
