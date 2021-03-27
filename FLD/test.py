# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:36:36 2020

@author: Samuel Osorio Gutiérrez

Programa que obtiene una matriz de transformación W a partir de los datos de entrenamiento para realizar
una reducción dimensional de cualquier entrada.
"""
import FLD
import numpy as np
from time import time
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

K=2 #reducción dimensional deseada, k <= min(C,D)-1
t_inicial=time()
W=FLD.train(features_train,targets_train,K, pLDA=True, plot=True,Terminal=True)
t_final=time()
print("\nTiempo de ejecución: " + str((t_final-t_inicial)*1000) + " ms")
#Y=np.matmul(features_train,W)#reduccion de las características de entrada a través de la matriz de transformación
