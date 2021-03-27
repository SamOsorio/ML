# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:28:50 2020

@author: Samuel Osorio Gutiérrez
"""

from FLD import FLD
from SFS import SFS
from LDA import LDA
from kNN import kNN
from MLP import MLP
from util import accuracy,normalization
from time import time
import numpy as np
np.random.seed(1234)

#base de datos de prueba
from sklearn.datasets import load_digits,load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split

iris=load_iris()
wine=load_wine()
digits=load_digits()
breast_cancer=load_breast_cancer()
#features_train, features_test, targets_train, targets_test=train_test_split(iris['data'],  iris['target'])
features_train, features_test, targets_train, targets_test=train_test_split(wine['data'],  wine['target'])
#features_train, features_test, targets_train, targets_test=train_test_split(digits['data'],  digits['target'])
#features_train, features_test, targets_train, targets_test=train_test_split(breast_cancer['data'],  breast_cancer['target'])
features_train=normalization(features_train)
features_test=normalization(features_test)
#Datos de los conjuntos
print("\x1b[1;33m"+ "Conjunto de entrenamiento con "+  str(features_train.shape[0]) + " muestras. ")
print("Conjunto de prueba con "+  str(features_test.shape[0]) + " muestras. ")
print("Número de clases: "+ str(max(targets_train)+1))
print("Dimensionalidad de los vectores de características: "+ str(features_train.shape[1]))


#SFS
K=9
t_inicial=time()
new_SFS_features_train, new_SFS_features_test=SFS.SFS(features_train, features_test, targets_train, targets_test, K, SFFS=False, name="LDA", pLDA=True)
t_final=time()
print("\x1b[1;37m"+"\nTiempo de ejecución de SFS a "+ str(K)+ " dimensiones: " + str((t_final-t_inicial)*1000) + " ms")

#FLD
K=4
t_inicial=time()
W=FLD.train(features_train,targets_train,K, pLDA=True, plot=True, Terminal=False)
t_final=time()
print("\nTiempo de ejecución de FLD a "+ str(K)+ " dimensiones: " + str((t_final-t_inicial)*1000) + " ms")

new_features_train=np.matmul(features_train,W)#reduccion de las características de entrada a través de la matriz de transformación
new_features_test=np.matmul(features_test,W)#reduccion de las características de entrada a través de la matriz de transformación

#LDA
print("\x1b[1;33m"+"\n*********************LDA*********************")

print("\nImplementación sin reducción de características:")

t_inicial=time()
S_inv,m_k,pi_k=LDA.train(features_train,targets_train, pLDA=True, Terminal=False)
t_final=time()
print("\x1b[1;37m"+"\nTiempo de entrenamiento: "  + str((t_final-t_inicial)*1000) + " ms")
prediction=LDA.classification(S_inv,m_k,pi_k,features_train)
acc=accuracy(prediction,targets_train)
print("\nExactitud en el conjunto de entrenamiento: " + str(acc*100) + " %")
t_inicial=time()
prediction=LDA.classification(S_inv,m_k,pi_k,features_test)
t_final=time()
acc=accuracy(prediction,targets_test)
print("\nExactitud en el conjunto de prueba: " + str(acc*100) + " %")
print("\nTiempo de clasificación (una muestra): " + str((t_final-t_inicial)*1000/features_test.shape[0]) + " ms")


print("\x1b[1;33m"+"\nImplementación con FLD:")

t_inicial=time()
S_inv,m_k,pi_k=LDA.train(new_features_train,targets_train, pLDA=True, Terminal=False)
t_final=time()
print("\x1b[1;37m"+"\nTiempo de entrenamiento: "  + str((t_final-t_inicial)*1000) + " ms")
prediction=LDA.classification(S_inv,m_k,pi_k,new_features_train)
acc=accuracy(prediction,targets_train)
print("\nExactitud en el conjunto de entrenamiento: " + str(acc*100) + " %")
t_inicial=time()
prediction=LDA.classification(S_inv,m_k,pi_k,new_features_test)
t_final=time()
acc=accuracy(prediction,targets_test)
print("\nExactitud en el conjunto de prueba: " + str(acc*100) + " %")
print("\nTiempo de clasificación (una muestra): " + str((t_final-t_inicial)*1000/features_test.shape[0]) + " ms")


#FLD
K=2
t_inicial=time()
W=FLD.train(new_SFS_features_train,targets_train,K, pLDA=True, plot=False, Terminal=False)
t_final=time()

new_features_train=np.matmul(new_SFS_features_train,W)#reduccion de las características de entrada a través de la matriz de transformación
new_features_test=np.matmul(new_SFS_features_test,W)#reduccion de las características de entrada a través de la matriz de transformación



print("\x1b[1;33m"+"\nImplementación con SFS-FLD:")

t_inicial=time()
S_inv,m_k,pi_k=LDA.train(new_features_train,targets_train, pLDA=True, Terminal=False)
t_final=time()
print("\x1b[1;37m"+"\nTiempo de entrenamiento: "  + str((t_final-t_inicial)*1000) + " ms")
prediction=LDA.classification(S_inv,m_k,pi_k,new_features_train)
acc=accuracy(prediction,targets_train)
print("\nExactitud en el conjunto de entrenamiento: " + str(acc*100) + " %")
t_inicial=time()
prediction=LDA.classification(S_inv,m_k,pi_k,new_features_test)
t_final=time()
acc=accuracy(prediction,targets_test)
print("\nExactitud en el conjunto de prueba: " + str(acc*100) + " %")
print("\nTiempo de clasificación (una muestra): " + str((t_final-t_inicial)*1000/features_test.shape[0]) + " ms")

#kNN
k=5
print("\x1b[1;33m"+"\n*********************kNN*********************")
print("k="+ str(k))

print("\nImplementación sin reducción de características:")

t_inicial=time()
prediction=kNN.classification(features_train, features_test, targets_train,k)
t_final=time()
acc=accuracy(prediction,targets_test)

print("\x1b[1;37m"+"\nExactitud en el conjunto de prueba: " + str(acc*100) + " %")
print("\nTiempo de clasificación con "+ str(features_train.shape[0]) +" distancias (una muestra): " + str((t_final-t_inicial)*1000/features_test.shape[0]) + " ms")


#FLD
K=2
t_inicial=time()
W=FLD.train(features_train,targets_train,K, pLDA=True, plot=False, Terminal=False)
t_final=time()

new_features_train=np.matmul(features_train,W)#reduccion de las características de entrada a través de la matriz de transformación
new_features_test=np.matmul(features_test,W)#reduccion de las características de entrada a través de la matriz de transformación


print("\x1b[1;33m"+"\nImplementación con FLD:")

t_inicial=time()
prediction=kNN.classification(new_features_train, new_features_test, targets_train,k)
t_final=time()
acc=accuracy(prediction,targets_test)

print("\x1b[1;37m"+"\nExactitud en el conjunto de prueba: " + str(acc*100) + " %")
print("\nTiempo de clasificación con "+ str(features_train.shape[0]) +" distancias (una muestra): " + str((t_final-t_inicial)*1000/features_test.shape[0]) + " ms")


#SFS
K=9
t_inicial=time()
new_SFS_features_train, new_SFS_features_test=SFS.SFS(features_train, features_test, targets_train, targets_test, K, SFFS=False, name="kNN", k=5)
t_final=time()

#FLD
K=2
t_inicial=time()
W=FLD.train(new_SFS_features_train,targets_train,K, pLDA=True, plot=False, Terminal=False)
t_final=time()

new_features_train=np.matmul(new_SFS_features_train,W)#reduccion de las características de entrada a través de la matriz de transformación
new_features_test=np.matmul(new_SFS_features_test,W)#reduccion de las características de entrada a través de la matriz de transformación



print("\x1b[1;33m"+"\nImplementación con SFS-FLD:")

t_inicial=time()
prediction=kNN.classification(new_features_train, new_features_test, targets_train,k)
t_final=time()
acc=accuracy(prediction,targets_test)

print("\x1b[1;37m"+"\nExactitud en el conjunto de prueba: " + str(acc*100) + " %")
print("\nTiempo de clasificación con "+ str(features_train.shape[0]) +" distancias (una muestra): " + str((t_final-t_inicial)*1000/features_test.shape[0]) + " ms")




#MLP
print("\x1b[1;33m"+"\n*********************MLP*********************")

hidden_layers=2
units=[4,6]
iterations=10
rate=5
function="sigmoid"
train="stochastic"
print("Número de capas ocultas: "+ str(hidden_layers))
print("Número de unidades en cada capa: ")
print(units)
print("Función de activación: "+ function)
print("Tipo de entrenamiento: "+ train)


print("\nImplementación sin reducción de características:")



t_inicial=time()
theta= MLP.train(features_train,targets_train,hidden_layers,units, iterations,rate, activation_function=function, Type=train ,Regularization=False, lamb=0.00001)
t_final=time()
print("\x1b[1;37m"+"\nTiempo de entrenamiento: "  + str((t_final-t_inicial)*1000) + " ms")
      
prediction=MLP.classification(features_train,theta,activation_function=function)
MLP_targets_train=MLP.targets_conv(targets_train,activation_function=function)
acc=MLP.accuracy(MLP_targets_train,prediction)
print("\nExactitud en el conjunto de entrenamiento: " + str(acc*100) + " %")
t_inicial=time()
prediction=MLP.classification(features_test,theta,activation_function=function)
t_final=time()
MLP_targets_test=MLP.targets_conv(targets_test,activation_function=function)
acc=MLP.accuracy(MLP_targets_test,prediction)
print("\nExactitud en el conjunto de prueba: " + str(acc*100) + " %")
t_final=time()
print("\nTiempo de clasificación (una muestra): " + str((t_final-t_inicial)*1000/features_test.shape[0]) + " ms")


print("\x1b[1;33m"+"\nImplementación con FLD:")

#FLD
K=2
t_inicial=time()
W=FLD.train(features_train,targets_train,K, pLDA=True, plot=False, Terminal=False)
t_final=time()

new_features_train=np.matmul(features_train,W)#reduccion de las características de entrada a través de la matriz de transformación
new_features_test=np.matmul(features_test,W)#reduccion de las características de entrada a través de la matriz de transformación

t_inicial=time()
theta= MLP.train(new_features_train,targets_train,hidden_layers,units, iterations,rate, activation_function=function, Type=train ,Regularization=False, lamb=0.00001)
t_final=time()
print("\x1b[1;37m"+"\nTiempo de entrenamiento: "  + str((t_final-t_inicial)*1000) + " ms")
      
prediction=MLP.classification(new_features_train,theta,activation_function=function)
acc=MLP.accuracy(MLP_targets_train,prediction)
print("\nExactitud en el conjunto de entrenamiento: " + str(acc*100) + " %")
t_inicial=time()
prediction=MLP.classification(new_features_test,theta,activation_function=function)
t_final=time()
acc=MLP.accuracy(MLP_targets_test,prediction)
print("\nExactitud en el conjunto de prueba: " + str(acc*100) + " %")
t_final=time()
print("\nTiempo de clasificación (una muestra): " + str((t_final-t_inicial)*1000/features_test.shape[0]) + " ms")


#SFS
K=9
t_inicial=time()
new_SFS_features_train, new_SFS_features_test=SFS.SFS(features_train, features_test,targets_train, targets_test, K, SFFS=False, name="MLP", hidden_layers=hidden_layers,units=units, iterations=iterations,rate=rate, activation_function=function)
t_final=time()

#FLD
K=2
t_inicial=time()
W=FLD.train(new_SFS_features_train,targets_train,K, pLDA=True, plot=False, Terminal=False)
t_final=time()

new_features_train=np.matmul(new_SFS_features_train,W)#reduccion de las características de entrada a través de la matriz de transformación
new_features_test=np.matmul(new_SFS_features_test,W)#reduccion de las características de entrada a través de la matriz de transformación


print("\x1b[1;33m"+"\nImplementación con SFS-FLD:")

t_inicial=time()
theta= MLP.train(new_features_train,targets_train,hidden_layers,units, iterations,rate, activation_function=function, Type=train ,Regularization=False, lamb=0.00001)
t_final=time()
print("\x1b[1;37m"+"\nTiempo de entrenamiento: "  + str((t_final-t_inicial)*1000) + " ms")
      
prediction=MLP.classification(new_features_train,theta,activation_function=function)
acc=MLP.accuracy(MLP_targets_train,prediction)
print("\nExactitud en el conjunto de entrenamiento: " + str(acc*100) + " %")
t_inicial=time()
prediction=MLP.classification(new_features_test,theta,activation_function=function)
t_final=time()
acc=MLP.accuracy(MLP_targets_test,prediction)
print("\nExactitud en el conjunto de prueba: " + str(acc*100) + " %")
t_final=time()
print("\nTiempo de clasificación (una muestra): " + str((t_final-t_inicial)*1000/features_test.shape[0]) + " ms")