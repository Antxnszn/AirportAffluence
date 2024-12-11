import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, classification_report
from datetime import datetime

# Ruta del archivo CSV
file_path = 'PassengersAffluency.csv'

# Cargar los datos
data = pd.read_csv(file_path)

# Crear una nueva columna 'data_dte' combinando 'Year' y 'Month'
data['data_dte'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1), errors='coerce')

# Eliminar filas con fechas inválidas
data = data[data['data_dte'].notna()]

# Selección de características para KMeans (usamos 'Month', 'Year' y 'Total')
features = data[['Month', 'Year', 'Total']].dropna()  # Eliminar filas con valores nulos

# Normalización de las características
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Aplicar KMeans (n_clusters=3 para baja, moderada, alta afluencia)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(features_scaled)

# Preparar datos para Random Forest
rf_features = data[['Month', 'Year', 'Cluster']]  # Características
target = data['Total']  # Etiqueta

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(rf_features, target, test_size=0.2, random_state=42)

# Entrenar el modelo Random Forest
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Realizar predicciones
y_pred = model_rf.predict(X_test)

# Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

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

# Predicción para una fecha futura
future_date = datetime(2024, 12, 25)  # Fecha futura
future_airport = 'JFK'  # Aeropuerto seleccionado

# Ejemplo de capacidad por aeropuerto
airport_capacity = {
    'LGA': 50000,  # Capacidad de LGA
    'JFK': 60000,  # Capacidad de JFK
    'DTW': 55000,  # Capacidad de DTW
    # Agregar las capacidades de otros aeropuertos
}

# Obtener la capacidad del aeropuerto seleccionado
future_capacity = airport_capacity.get(future_airport, 50000)  # Valor por defecto

# Predecir la afluencia de pasajeros para esa fecha
future_features = pd.DataFrame({
    'Month': [future_date.month],
    'Year': [future_date.year],
    'Cluster': [kmeans.predict(scaler.transform([[future_date.month, future_date.year, 0]]))[0]]  # Predicción de cluster
})

predicted_passengers = model_rf.predict(future_features)

# Clasificación de la afluencia según la capacidad del aeropuerto
if predicted_passengers > future_capacity * 0.8:
    afluencia = 'Alta'
elif predicted_passengers > future_capacity * 0.5:
    afluencia = 'Moderada'
else:
    afluencia = 'Baja'

print(f"La afluencia esperada para {future_date.date()} en el aeropuerto {future_airport} es: {afluencia}")
