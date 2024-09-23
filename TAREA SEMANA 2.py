import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC


#Carga de datos
data = pd.read_table('/Users/angel/Documents/maestria/4. Introduction a la ciencia de datos/clase 1/First python prject/clase 4/dataset1.csv', delimiter=',')
data.reset_index(inplace=True)
data.columns = ['X1', 'X2', 'y']

#Copia de la data
df = data.copy()
print(df.head(),'\n')

#Declarando variables
X1=df.iloc[:,0]
X2=df.iloc[:,1]
X=np.column_stack((X1,X2))
y=df.iloc[:,2]

#PARTE A (i)
#Grafico 2D
plt.scatter(X1[y==1], X2[y==1], color='blue', marker = '+', label='Clase 1',)
plt.scatter(X1[y==-1], X2[y==-1], color='red', marker = 'o', label='Clase -1', s=10)
plt.title('Visualización de los datos +1 y -1')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='lower right')
plt.show()


#PARTE A (ii)
#Dividiendo la data en 80% datos de entrenamiento y 20% datos de prueba
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2, train_size=0.8,random_state=4)
print('Train Set 80%: ', x_train.shape, y_train.shape)
print('Test Set 20% : ', x_test.shape, y_test.shape, '\n')

LR=LogisticRegression()
#Entrenando el modelo logistico
LR.fit(x_train, y_train)

#Parametros del modelo logistico entrenado
print('Los resultados para la regresion logistica son: ')
print('La pendiente es: ',LR.coef_[0])
print('La intercepción es: ',LR.intercept_)
prediccion = LR.predict(x_train)
precision = LR.score(x_train, y_train)
print('La precisión es: ',precision,'\n')

#PARTE A (iii)
#Grafico valores reales
plt.scatter(x_train[y_train==1,0], x_train[y_train==1,1], color='blue', marker = '+', label='Verdadero 1',)
plt.scatter(x_train[y_train==-1,0], x_train[y_train==-1,1], color='red', marker = 'o', label='Verdadero -1', s=10)

#Grafico valores predichos
plt.scatter(x_train[prediccion==1,0],x_train[prediccion==1,1],color='green', marker='x', label= 'Predicho 1' , alpha=0.5)
plt.scatter(x_train[prediccion==-1,0], x_train[prediccion==-1,1],  color='orange',marker='*',label= 'Predicho -1', alpha=0.5)

# Frontera de decision (X2 = -(coef[0] * X1 + intercept) / coef[1])
x_valores = np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), 100)
front_desicion = -(LR.coef_[0][0] * x_valores + LR.intercept_[0]) / LR.coef_[0][1]
plt.plot(x_valores, front_desicion, 'k--', label='Frontera de decision')

plt.title('Visualización de los datos de entrenamiento con la frontera de decision')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='lower right')
plt.show()


#PARTE B (i)
#Modelo SVM
C_valores = [0.001, 1, 100]

for C in C_valores:
    svm_model = LinearSVC(max_iter=10000, C=C)
    svm_model.fit(x_train, y_train)

    svm_prediccion = svm_model.predict(x_test)
    svm_score = svm_model.score(x_test, y_test)
    cm = confusion_matrix(y_test, svm_prediccion)
    clasificacion = classification_report(y_test, svm_prediccion)
    svm_coef = svm_model.coef_[0]
    smv_intercept = svm_model.intercept_


    print(f"Precisión con C={C}:\n{svm_score:}")
    print(f"Confusión con C={C}:\n{cm}")
    print(f"Clasificacion con C={C}:\n{clasificacion}")
    print(f"La pendiente con C={C}:\n{svm_coef}")
    print(f"la intercepcion con C={C}:\n{smv_intercept}\n")

    # PARTE B (ii)
    #Grafico valores reales
    plt.scatter(x_test[y_test==1,0], x_test[y_test==1,1], color='blue', marker = '+', label='Verdadero 1',)
    plt.scatter(x_test[y_test==-1,0], x_test[y_test==-1,1], color='red', marker = 'o', label='Verdadero -1', s=10)

    #Grafico valores predichos SVM
    plt.scatter(x_test[svm_prediccion == 1, 0], x_test[svm_prediccion == 1, 1], color='green', marker='x',label='Predicho 1', alpha=0.5)
    plt.scatter(x_test[svm_prediccion == -1, 0], x_test[svm_prediccion == -1, 1], color='orange', marker='*',label='Predicho -1', alpha=0.5)

    #Grafico frontera de decision SVM
    x_valores_svm = np.linspace(x_test[:, 0].min(), x_test[:, 0].max(), 100)
    front_desicion_svm = -(svm_model.coef_[0][0] * x_valores_svm + svm_model.intercept_[0]) / svm_model.coef_[0][1]
    plt.plot(x_valores_svm, front_desicion_svm, 'k--', label='Frontera de decision SVM')

    plt.title(f"Frontera de decision SVM para C={C}")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend(loc='lower right')
    plt.show()


