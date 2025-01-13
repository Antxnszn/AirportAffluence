from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Selección de características (features) y etiqueta (target)
X = data[['mes', 'dia', 'aeropuerto', 'cluster']]  # Variables independientes
y = data['num_pasajeros']  # Variable dependiente (lo que quieres predecir)

# Convertir variables categóricas (si las hay) a numéricas
X = pd.get_dummies(X, drop_first=True)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Hacer predicciones
y_pred = rf_model.predict(X_test)

# Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
print(f'Error Medio Absoluto: {mae}')
