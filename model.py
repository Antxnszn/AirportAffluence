import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo proporcionado
file_path = 'PassengersAffluency.csv'
data = pd.read_csv(file_path)

# Limpiar columnas irrelevantes o vacías
data = data.drop(columns=['Unnamed: 9'], errors='ignore')

# Asegurarse de que las columnas necesarias sean numéricas
data['OcupancyPercentage'] = pd.to_numeric(data['OcupancyPercentage'], errors='coerce')

# Si la columna Concurrence es categórica, convertirla a variables dummy
if data['Concurrence'].dtype == 'object':
    data = pd.get_dummies(data, columns=['Concurrence'], prefix='Concurrence')

# Imprimir diagnóstico inicial
print("Datos cargados, primeras filas:")
print(data.head())

# Eliminar filas con valores faltantes en las columnas relevantes
features = data[['Month', 'Year', 'Total'] + [col for col in data.columns if 'Concurrence_' in col] + ['OcupancyPercentage']]
print("Número de filas antes de dropna:", features.shape[0])
features = features.dropna()
print("Número de filas después de dropna:", features.shape[0])

# Verificar si el DataFrame está vacío
if features.empty:
    raise ValueError("El DataFrame 'features' está vacío después de aplicar dropna. Verifica los datos de entrada.")

# Normalizar las características
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Aplicar KMeans con 3 clústeres (baja, moderada, alta afluencia)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(features_scaled)

# Preparar datos para Random Forest, ahora incluyendo 'Concurrence' y 'OcupancyPercentage'
rf_features = data[['Month', 'Year', 'Cluster'] + [col for col in data.columns if 'Concurrence_' in col] + ['OcupancyPercentage']]
target = data['Total']  # Etiqueta

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(rf_features, target, test_size=0.2, random_state=42)

# Entrenar el modelo Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Realizar predicciones y calcular métricas de desempeño
y_pred = model_rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# Imprimir resultados de las métricas
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Función para categorizar afluencia
def categorize_affluence(value, capacity=55000):
    if value > capacity * 0.8:
        return 'Alta'
    elif value > capacity * 0.5:
        return 'Moderada'
    else:
        return 'Baja'

# Categorizar valores reales y predichos
y_test_categorized = y_test.apply(categorize_affluence)
y_pred_categorized = pd.Series(y_pred).apply(categorize_affluence)

# Matriz de confusión
conf_matrix = confusion_matrix(y_test_categorized, y_pred_categorized, labels=['Baja', 'Moderada', 'Alta'])

# Graficar la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Baja', 'Moderada', 'Alta'], yticklabels=['Baja', 'Moderada', 'Alta'])
plt.title('Matriz de Confusión')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.show()

# Reporte de clasificación
print("Reporte de Clasificación:")
print(classification_report(y_test_categorized, y_pred_categorized))

# Graficar la relación entre valores reales y predichos
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Relación entre valores reales y predichos')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.show()

# Histograma para comparar distribuciones
plt.figure(figsize=(8, 6))
sns.histplot(y_test, color='blue', label='Reales', kde=True)
sns.histplot(y_pred, color='orange', label='Predichos', kde=True)
plt.title('Distribución de Valores Reales y Predichos')
plt.legend()
plt.show()

# Definir capacidades por aeropuerto
airport_capacity = {
    'LGA': 50000,  # Capacidad de LGA
    'JFK': 60000,  # Capacidad de JFK
    'DTW': 55000,  # Capacidad de DTW
    # Agregar las capacidades de otros aeropuertos
}

# Solicitar datos al usuario
print("=== Predicción de Afluencia de Pasajeros ===")
date_input = input("Introduce la fecha futura en formato YYYY-MM-DD: ")
future_airport = input("Introduce el código del aeropuerto (ejemplo: LGA, JFK, DTW): ").strip().upper()

# Validar y convertir la fecha ingresada
try:
    future_date = datetime.strptime(date_input, "%Y-%m-%d")
except ValueError:
    print("La fecha introducida no es válida. Usa el formato YYYY-MM-DD.")
    exit()

# Obtener la capacidad del aeropuerto seleccionado
future_capacity = airport_capacity.get(future_airport, 50000)  # Valor por defecto si no está en el diccionario

# Construir las características para la predicción
future_features = pd.DataFrame({
    'Month': [future_date.month],
    'Year': [future_date.year],
    'Cluster': [kmeans.predict(scaler.transform([[future_date.month, future_date.year, 0] + [0] * len([col for col in data.columns if 'Concurrence_' in col]) + [0]]))[0]],
    **{col: [0] for col in [col for col in data.columns if 'Concurrence_' in col]},
    'OcupancyPercentage': [0]  # Ajustar según valores futuros esperados
})

# Realizar la predicción
predicted_passengers = model_rf.predict(future_features)

# Clasificar la afluencia según la capacidad
if predicted_passengers > future_capacity * 0.8:
    afluencia = 'Alta'
elif predicted_passengers > future_capacity * 0.5:
    afluencia = 'Moderada'
else:
    afluencia = 'Baja'

# Mostrar el resultado
print(f"\nLa afluencia esperada para {future_date.date()} en el aeropuerto {future_airport} es: {afluencia}")
