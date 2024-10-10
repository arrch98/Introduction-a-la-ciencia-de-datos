import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score, train_test_split

# Cargar y limpiar los datos
file_path = '/Users/angel/Documents/maestria/4. Introduction a la ciencia de datos/clase 1/First python prject/clase 9/week3.csv'
data = pd.read_csv(file_path, delimiter=',')

print(f"Número de columnas: {data.shape[1]}")

# Asegurar de que el número de columnas coincide con los nombres asignados
if data.shape[1] == 3:
    data.columns = ['Feature_1', 'Feature_2', 'Target']
elif data.shape[1] == 2:
    data.columns = ['Feature_1', 'Feature_2']
else:
    print("El número de columnas no coincide con el esperado.")

print(data.head())

# renombramos las columnas
data.columns = ['Feature_1', 'Feature_2', 'Target']

# Convertir a tipo numérico
data = data.apply(pd.to_numeric)

# Extraer las características
X = data[['Feature_1', 'Feature_2']].values
y = data['Target'].values


# (i) (a) Gráfico de dispersión 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, c='b', marker='o')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.set_title('Grafico de dispersion 3D')
plt.show()

# (i) (b) Crear características polinómicas y entrenar Lasso con diferentes valores de C

# Definir los grados polinómicos que se van a utilizar
degrees = [1, 2, 3, 4, 5]

# Definir diferentes valores de C
C_values = [0.1, 1, 10, 1000]
alphas = [1 / (2 * C) for C in C_values]  # Lasso, alpha = 1 / (2C)

lasso_models = {}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Entrenar Lasso para diferentes grados y valores de C
for degree in degrees:
    print(f"\n Entrenando modelos para grado polinómico {degree}...")

    poly_1 = PolynomialFeatures(degree=degree)
    X_train_poly_1 = poly_1.fit_transform(X_train)
    X_test_poly = poly_1.transform(X_test)

    coefficients = {}
    # Entrenar modelos para diferentes valores de C
    for alpha, C in zip(alphas, C_values):
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(X_train_poly_1, y_train)
        lasso_models[(degree, C)] = model
        coefficients[C] = model.coef_

    coef_df = pd.DataFrame(coefficients, index=poly_1.get_feature_names_out(['Feature_1', 'Feature_2']))
    print(f"\n Coeficientes de Lasso para grado {degree}:")
    print(coef_df)

# (i) (c) Predicciones en una cuadrícula de valores

# Generar una cuadrícula de valores para las características
grid = np.linspace(-5, 5, 50)
Xtest = np.array([[i, j] for i in grid for j in grid])

# Graficar las predicciones para cada grado polinómico y modelo Lasso
fig = plt.figure(figsize=(34, 22))

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_test_poly = poly.fit_transform(Xtest)

    for idx, C in enumerate(C_values, 1):
        ax = fig.add_subplot(len(degrees), len(C_values), (degree - 1) * len(C_values) + idx, projection='3d')

        # Realizar predicciones
        y_pred = lasso_models[(degree, C)].predict(X_test_poly)

        X1_grid, X2_grid = np.meshgrid(grid, grid)
        y_pred_grid = y_pred.reshape(X1_grid.shape)

        # Graficar
        ax.plot_surface(X1_grid, X2_grid, y_pred_grid, color='r', alpha=0.7)

        ax.scatter(X[:, 0], X[:, 1], y, c='b', marker='o', label='Datos de entrenamiento')

        # Etiquetas y título
        ax.set_title(f'Grado {degree} - Predicciones Lasso (C={C})')
        ax.set_xlabel('Característica 1')
        ax.set_ylabel('Característica 2')
        ax.set_zlabel('Objetivo')
        ax.legend()

plt.tight_layout()
plt.show()

# (i) (e) Predicciones de Ridge en una cuadrícula de valores

# Entrenar modelos Ridge para diferentes grados polinómicos y valores de C
ridge_models = {}

for degree in degrees:

    # Crear características polinómicas para el grado actual
    poly_2 = PolynomialFeatures(degree=degree)
    X_train_poly_2 = poly_2.fit_transform(X_train)
    X_test_poly_2 = poly_2.transform(X_test)

    coefficients = {}

    # Entrenar modelos Ridge para diferentes valores de C
    for alpha, C in zip(alphas, C_values):
        model = Ridge(alpha=alpha, max_iter=10000)
        model.fit(X_train_poly_2, y_train)
        ridge_models[(degree, C)] = model
        coefficients[C] = model.coef_

# (ii) Validación cruzada para Lasso
C_range = np.logspace(-4, 4, 10)
alphas = [1 / (2 * C) for C in C_range]

mean_scores = []
std_scores = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    scores = cross_val_score(lasso, X_train_poly_2, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_scores.append(-scores.mean())
    std_scores.append(scores.std())

# Graficar el error promedio y la desviación estándar frente a C
plt.figure(figsize=(10, 6))
plt.errorbar(C_range, mean_scores, yerr=std_scores, capsize=5)
plt.xscale('log')
plt.xlabel('Valores de C')
plt.ylabel('Error cuadrático medio')
plt.title('Validación cruzada de Lasso: Error vs C')
plt.show()

# (ii) Validación cruzada para Ridge
mean_scores_ridge = []
std_scores_ridge = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha, max_iter=10000)
    scores_ridge = cross_val_score(ridge, X_train_poly_2, y_train, cv=5, scoring='neg_mean_squared_error')
    mean_scores_ridge.append(-scores_ridge.mean())
    std_scores_ridge.append(scores_ridge.std())

# Gráfico comparativo entre Lasso y Ridge
plt.figure(figsize=(10, 6))

plt.errorbar(C_range, mean_scores, yerr=std_scores, label='Lasso', capsize=5, color='blue')
plt.errorbar(C_range, mean_scores_ridge, yerr=std_scores_ridge, label='Ridge', capsize=5, color='orange')

plt.xscale('log')
plt.xlabel('Valores C')
plt.ylabel('R cuadrado medio')
plt.title('Comparacion validacion cruzada Lasso vs Ridge')
plt.legend()
plt.show()


ridge_models_1 = {}

# Entrenar Ridge para diferentes grados polinómicos y valores de C
for degree in degrees:
    print(f"\n Entrenando modelos Ridge para grado polinómico {degree}...")

    # Crear características polinómicas para el grado actual
    poly_3 = PolynomialFeatures(degree=degree)
    X_train_poly_3 = poly_3.fit_transform(X_train)
    X_test_poly_3 = poly_3.transform(X_test)

    ridge_coefficients = {}

    # Entrenar modelos Ridge para diferentes valores de C
    for alpha, C in zip(alphas, C_values):
        model = Ridge(alpha=alpha, max_iter=10000)
        model.fit(X_train_poly_3, y_train)
        ridge_models_1[(degree, C)] = model
        ridge_coefficients[C] = model.coef_

    # Mostrar coeficientes con print o visualización estándar
    ridge_coef_df = pd.DataFrame(ridge_coefficients, index=poly_3.get_feature_names_out(['Feature_1', 'Feature_2']))
    print(f"\n Coeficientes de Ridge para grado {degree}:")
    print(ridge_coef_df)

# Graficar el error promedio y desviación estándar frente a C para Ridge
plt.figure(figsize=(10, 6))
plt.errorbar(C_range, mean_scores_ridge, yerr=std_scores_ridge, capsize=5, color='orange')
plt.xscale('log')
plt.xlabel('Valores de C')
plt.ylabel('Error cuadrático medio')
plt.title('Validación cruzada de Ridge: Error vs C')
plt.show()

  #  Generar una cuadrícula devalores para las características
grid = np.linspace(-5, 5, 50)
Xtest2 = np.array([[i, j] for i in grid for j in grid])

fig = plt.figure(figsize=(34, 22))

for degree in degrees:
    poly_1 = PolynomialFeatures(degree=degree)
    X_test_poly_1 = poly_1.fit_transform(Xtest2)

    for idx, C in enumerate(C_values, 1):
        ax = fig.add_subplot(len(degrees), len(C_values), (degree - 1) * len(C_values) + idx, projection='3d')

        # Realizar predicciones con Ridge
        y_pred2 = ridge_models[(degree, C)].predict(X_test_poly_1)

        X1_grid_2, X2_grid_2 = np.meshgrid(grid, grid)
        y_pred2_grid = y_pred2.reshape(X1_grid_2.shape)

        # Graficar
        ax.plot_surface(X1_grid_2, X2_grid_2, y_pred2_grid, color='r', alpha=0.7)

        # Graficar los datos de entrenamiento
        ax.scatter(X[:, 0], X[:, 1], y, c='b', marker='o', label='Datos de entrenamiento')

        # Etiquetas y título
        ax.set_title(f'Grado {degree} - Predicciones Ridge (C={C})')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Target')
        ax.legend()

plt.tight_layout()
plt.show()





