# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:46:08 2020

@author: Samuel Osorio Gutiérrez
"""

import MLP
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

hidden_layers=2
units=[6,5]
iterations=8
rate=10
function="sigmoid"

features_train=MLP.normalization(features_train)
features_test=MLP.normalization(features_test)
t_inicial=time()
theta= MLP.train(features_train,targets_train,hidden_layers,units, iterations,rate, activation_function=function, Type="stochastic",Regularization=False, lamb=0.00001)
t_final=time()
print("\nTiempo de entrenamiento: "  + str((t_final-t_inicial)*1000) + " ms")
      
prediction=MLP.classification(features_train,theta,activation_function=function)
targets_train=MLP.targets_conv(targets_train,activation_function=function)
acc=MLP.accuracy(targets_train,prediction)
print("\nExactitud en el conjunto de entrenamiento: " + str(acc*100) + " %")
t_inicial=time()
prediction=MLP.classification(features_test,theta,activation_function=function)
t_final=time()
targets_test=MLP.targets_conv(targets_test,activation_function=function)
acc=MLP.accuracy(targets_test,prediction)
print("\nExactitud en el conjunto de prueba: " + str(acc*100) + " %")
t_final=time()
print("\nTiempo de clasificación (una muestra): " + str((t_final-t_inicial)*1000/features_test.shape[0]) + " ms")