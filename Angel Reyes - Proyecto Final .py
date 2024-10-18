import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Cargar datos
data = pd.read_csv('/Users/angel/Documents/maestria/4. Introduction a la ciencia de datos/clase 1/First python prject/Proyecto final/Dataset Covid19 Panama 10-05-2020.csv')


# 1. Comprension del problema y el dataset
print(data.info(),'\n')
print(data.describe(),'\n')


# Grafica de dispersion la data inicial
plt.figure(figsize=(10, 6))
plt.scatter(data['LONG'], data['LAT'], c=data['CANTIDAD'], cmap='viridis', alpha=0.7)
plt.colorbar(label='Cantidad de Casos')
plt.title('Distribución Geográfica de los Casos')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.grid(True)
plt.show()


# 2.Preparación de los datos
# Verificamos valores faltantes
print("Valores faltantes en el dataset:\n", data.isnull().sum())


#imputacion de valores 0
columnas_imputar = ['CANTIDAD','HOSPITALIZADO', 'AISLAMIENTO_DOMICILIARIO', 'FALLECIDO', 'UCI', 'RECUPERADO']
data[columnas_imputar] = data[columnas_imputar].fillna(0)

# Verificamos valores faltantes nuevamente
print("Valores faltantes en el dataset:\n", data.isnull().sum())

plt.figure(figsize=(10, 6))
plt.scatter(data['LONG'], data['LAT'], c=data['CANTIDAD'], cmap='viridis', alpha=0.7)
plt.colorbar(label='Cantidad de Casos')
plt.title('Distribución Geográfica de los Casos')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.grid(True)
plt.show()

# 3. Feature Engineering
# agregamos 3 caracteristicas nuevas
data['Hospitalizado_Fallecido'] = data.apply(lambda x: x['HOSPITALIZADO'] / x['FALLECIDO'] if x['FALLECIDO'] != 0 else 0, axis=1)

data['Recuperado_aislamiento'] = data.apply(lambda x: x['RECUPERADO'] / x['AISLAMIENTO_DOMICILIARIO'] if x['AISLAMIENTO_DOMICILIARIO'] != 0 else 0, axis=1)

data['Casos_criticos'] = data['UCI'] + data['HOSPITALIZADO']

print(data.head)

# 4. División del dataset y Validación Cruzada
# Seleccionamos variables
features = ['CANTIDAD', 'HOSPITALIZADO', 'AISLAMIENTO_DOMICILIARIO', 'FALLECIDO', 'UCI', 'RECUPERADO',
            'Hospitalizado_Fallecido', 'Recuperado_aislamiento', 'Casos_criticos']

# Separamos chiriqui y bocas del toro del dataset
data_chiriqui = data[data['PROVINCIA'] == 'CHIRIQUÍ']
data_bocas = data[data['PROVINCIA'] == 'BOCAS DEL TORO']

# Variables target
# Chiriqui
X_chiriqui = data_chiriqui[features]
y_chiriqui = data_chiriqui['CANTIDAD']
X_train_chiriqui, X_test_chiriqui, y_train_chiriqui, y_test_chiriqui = train_test_split(X_chiriqui, y_chiriqui, test_size=0.3, random_state=42)

# Bocas del Toro
X_bocas = data_bocas[features]
y_bocas = data_bocas['CANTIDAD']
X_train_bocas, X_test_bocas, y_train_bocas, y_test_bocas = train_test_split(X_bocas, y_bocas, test_size=0.3, random_state=42)

# Entrenado de modelos y validacion cruzada
linear_model = LinearRegression()
rf_model = RandomForestRegressor(random_state=42)

# Definimos k-fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Chiriquí
cv_scores_chiriqui_linear = cross_val_score(linear_model, X_train_chiriqui, y_train_chiriqui, cv=kf, scoring='neg_mean_squared_error')
cv_scores_chiriqui_rf = cross_val_score(rf_model, X_train_chiriqui, y_train_chiriqui, cv=kf, scoring='neg_mean_squared_error')

# Bocas del Toro
cv_scores_bocas_linear = cross_val_score(linear_model, X_train_bocas, y_train_bocas, cv=kf, scoring='neg_mean_squared_error')
cv_scores_bocas_rf = cross_val_score(rf_model, X_train_bocas, y_train_bocas, cv=kf, scoring='neg_mean_squared_error')


# imprimir resultados
print('Validacion cruzada Regresion lineal Chiriqui:',cv_scores_chiriqui_linear)
print('\n Validacion cruzada Regresion random forest Chiriqui:',cv_scores_chiriqui_rf)
print('\n Validacion cruzada Regresion lineal Bocas:',cv_scores_bocas_linear)
print('\n Validacion cruzada Regresion random forest Bocas:',cv_scores_bocas_rf)

# Grafico validacion cruzada chiriqui
plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), -cv_scores_chiriqui_linear, label='Linear Regression - Chiriquí', marker='o', color='green')
plt.plot(range(1, 6), -cv_scores_chiriqui_rf, label='Random Forest - Chiriquí', marker='o', color='red')
plt.title('Validación Cruzada - Chiriquí (Error vs Modelo)')
plt.xlabel('Fold')
plt.ylabel('Error cuadrado')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Grafico validacion cruzada Bocas del Toro
plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), -cv_scores_bocas_linear, label='Linear Regression - Bocas del Toro', marker='x', color='blue')
plt.plot(range(1, 6), -cv_scores_bocas_rf, label='Random Forest - Bocas del Toro', marker='x', color='orange')
plt.title('Validación Cruzada - Bocas del Toro (Error vs Modelo)')
plt.xlabel('Fold')
plt.ylabel('Error cuadrado')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 5. Modelado
# Entrenamos los modelos

### Regresion lineal
# Chiriqui
linear_model.fit(X_train_chiriqui, y_train_chiriqui)
y_pred_chiriqui_linear = linear_model.predict(X_test_chiriqui)

# Bocas del toro
linear_model.fit(X_train_bocas, y_train_bocas)
y_pred_bocas_linear = linear_model.predict(X_test_bocas)

### Random Forest
# Chiriqui
rf_model.fit(X_train_chiriqui, y_train_chiriqui)
y_pred_chiriqui_rf = rf_model.predict(X_test_chiriqui)

# Bocas del toro
rf_model.fit(X_train_bocas, y_train_bocas)
y_pred_bocas_rf = rf_model.predict(X_test_bocas)


# 6. Metricas: MAE, MSE, RMSE, R2
resultados = pd.DataFrame({
    'Modelo': ['Linear Regression - Chiriquí', 'Random Forest - Chiriquí',
              'Linear Regression - Bocas del Toro', 'Random Forest - Bocas del Toro'],
    'MAE': [
        mean_absolute_error(y_test_chiriqui, y_pred_chiriqui_linear),
        mean_absolute_error(y_test_chiriqui, y_pred_chiriqui_rf),
        mean_absolute_error(y_test_bocas, y_pred_bocas_linear),
        mean_absolute_error(y_test_bocas, y_pred_bocas_rf)],
    'MSE': [
        mean_squared_error(y_test_chiriqui, y_pred_chiriqui_linear),
        mean_squared_error(y_test_chiriqui, y_pred_chiriqui_rf),
        mean_squared_error(y_test_bocas, y_pred_bocas_linear),
        mean_squared_error(y_test_bocas, y_pred_bocas_rf)],
    'RMSE': [
        np.sqrt(mean_squared_error(y_test_chiriqui, y_pred_chiriqui_linear)),
        np.sqrt(mean_squared_error(y_test_chiriqui, y_pred_chiriqui_rf)),
        np.sqrt(mean_squared_error(y_test_bocas, y_pred_bocas_linear)),
        np.sqrt(mean_squared_error(y_test_bocas, y_pred_bocas_rf))],
    'R2 Score': [
        r2_score(y_test_chiriqui, y_pred_chiriqui_linear),
        r2_score(y_test_chiriqui, y_pred_chiriqui_rf),
        r2_score(y_test_bocas, y_pred_bocas_linear),
        r2_score(y_test_bocas, y_pred_bocas_rf)] })


# Resultados Chiriquí
resultados_chiriqui = resultados[resultados['Modelo'].str.contains('Chiriquí')]
print("\nMetricas Chiriquí:\n", resultados_chiriqui)

# Resultados Bocas del Toro
resultados_bocas = resultados[resultados['Modelo'].str.contains('Bocas del Toro')]
print("\nMetricas Bocas del Toro:\n", resultados_bocas)


# Resultados
plt.figure(figsize=(14, 6))

# Predicciones Chiriquí - Regresion lineal vs Random Forest
plt.subplot(1, 2, 1)
plt.plot(y_test_chiriqui.values, label='Actual Values', color='blue')
plt.plot(range(len(y_pred_chiriqui_linear)), y_pred_chiriqui_linear, label='Linear Regression Predictions', color='green', linestyle='--')
plt.plot(range(len(y_pred_chiriqui_rf)), y_pred_chiriqui_rf, label='Random Forest Predictions', color='red', linestyle=':')
plt.title('Predicciones para Chiriquí')
plt.xlabel('Muestras de Prueba')
plt.ylabel('Valor de Recuperados')
plt.legend(fontsize='small')
plt.grid(True)

# Predicciones Bocas del Toro  - Regresion lineal vs Random Forest
plt.subplot(1, 2, 2)
plt.plot(y_test_bocas.values, label='Actual Values', color='blue')
plt.plot(range(len(y_pred_bocas_linear)), y_pred_bocas_linear, label='Linear Regression Predictions', color='green', linestyle='--')
plt.plot(range(len(y_pred_bocas_rf)), y_pred_bocas_rf, label='Random Forest Predictions', color='red', linestyle=':')
plt.title('Predicciones para Bocas del Toro')
plt.xlabel('Muestras de Prueba')
plt.ylabel('Valor de Fallecidos')
plt.legend(fontsize='small')
plt.grid(True)
plt.show()