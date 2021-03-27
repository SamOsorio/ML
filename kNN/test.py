# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 00:08:55 2020

@author: Samuel Osorio Gutiérrez
"""
import kNN
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
t_inicial=time()
prediction=kNN.classification(features_train, features_test, targets_train,k=5)
t_final=time()
acc=kNN.accuracy(prediction,targets_test)
print("Exactitud de: " + str(acc*100) + " %")
print("\nTiempo de clasificación con "+ str(features_train.shape[0]) +" distancias (una muestra): " + str((t_final-t_inicial)*1000/features_test.shape[0]) + " ms")

#kNN.fit(15, features_train, features_test, targets_train, targets_test)
