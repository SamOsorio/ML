# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:56:31 2020

@author: Samuel Osorio Gutiérrez
"""

import SFS
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


features_train=SFS.normalization(features_train)
features_test=SFS.normalization(features_test)

t_inicial=time()
new_features_train, new_features_test=SFS.SFS(features_train, features_test, targets_train, targets_test, K=9, SFFS=False, name="LDA", pLDA=True)
t_final=time()
print("\nTiempo de ejecución: " + str((t_final-t_inicial)*1000) + " ms")