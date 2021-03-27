# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:25:15 2020

@author: Samuel Osorio Gutiérrez
"""

import numpy as np

def indexClass(targets_train):
    """
    Realiza la identificacion de los indices correspondientes a cada clase dentro de los datos de entrenamiento

    Parameters
    ----------
    targets_train : array
        vector que contiene las clases dentro del conjunto de entrenamiento

    Returns
    -------
    C : int
        indica el numero máximo de clases dentro del conjunto de entrenamiento
    OmegaC : list of list
        lista de listas que contiene los indices correspondientes a cada clase del conjunto de entrenamiento
    """
    C=max(targets_train) #se obtiene el maximo numero de clases
    N=targets_train.shape[0]
    OmegaC=[]
    for c in range(C+1): #se busca cada clase dentro del arreglo para agregar sus indices a una lista
        listC=[]
        for n in range (N):
                if targets_train[n] == c:
                    listC.append(n)
        OmegaC.append(listC)  #se crea una lista con las listas de indices de cada clase
    return C,OmegaC 

def distance(features_train, features_entry):
    """
    Se calcula la distancia de cada entrada respecto a los vectores de entrenamiento

    Parameters
    ----------
    features_train : array
        Matriz con vectores de características de entrenamiento
    features_entry : array
        Matriz con vectores de características a clasificar

    Returns
    -------
    d : array
        Matriz con distancais de todos los vectores de características respecto de los vectores a clasificar
    N_entry : int
        Número de vectores de características de entrada
    """
    N_entry=features_entry.shape[0]
    N_train=features_train.shape[0]
    d=np.zeros((N_train,N_entry)) #se crea la matriz de distancias

    for n_entry in range(N_entry):
        z=features_entry[n_entry,:]#vector z con el que se sacaran las distancias
        for n_train in range(N_train): #distancias con cada uno de los datos de entrenamiento
            xn=features_train[n_train,:]
            d[n_train, n_entry]= np.matmul(z-xn,z-xn)#al hacer una multiplicacion de dos arrays de 1D se obtiene la distancia cuadrada 
    return d,N_entry

def distanceMin(d,k,N_entry):
    """
    Se busca dentro del vector de distanas las k más pequeñas junto con sys indices

    Parameters
    ----------
    d : array
        Matriz con distancais de todos los vectores de características respecto de los vectores a clasificar
    k : int
        Vecinos más cercanos al vector de características a clasificar
    N_entry : int
        Número de vectores de características de entrada

    Returns
    -------
    kindex : array
        Indices de las k distancias más cercanas en orden
    kdistance: array
        k distancias más cercanas en orden
    """
    listIndex=[] #se crea una lista con los indices de las distancias más cercanas
    d=np.transpose(d)
    d=list(d) #se convierte el vector de distancias en una lista de listas
    kdistance=np.zeros([N_entry,k]) #se crea la matriz con las distancias mas cercanas
    kindex=np.zeros([N_entry,k]) #se crea la matriz con las distancias mas cercanas
    for vector in range(N_entry):
        d_temp=list(d[vector])
        dindex_temp=list(d[vector])
        for i in range(k):#se busca el valor máximo, se guarda su indice, se elimina de la lista y se guarda el vector correspondiente 
            minValue=min(d_temp)
            index=d_temp.index(minValue)
            listIndex.append(index)
            d_temp.pop(index)
            kindex[vector,i]=dindex_temp.index(minValue) #se busca el indice en el vector original
            kdistance[vector,i]=minValue 
    return kindex,kdistance 

def target(kindex, kdistance, targets_train ,N_entry,k):
    """
    Se busca la clase más común dentro de los k vecinos y se le asigna al vector de entrada

    Parameters
    ----------
    kindex : array
        Indices de las k distancias más cercanas en orden
    kdistance: array
        k distancias más cercanas en orden
    targets_train : array
        vector que contiene las clases dentro del conjunto de entrenamiento
    N_entry : int
        Número de vectores de características de entrada
    k : int
        Vecinos más cercanos al vector de características a clasificar

    Returns
    -------
    prediction : array
        Vector con las clases predichas
    """
    kindex=list(kindex) #una lista con las listas de indices de los vecinos más cercanos respecto a cada entrada
    prediction=np.zeros(N_entry,) #arreglo donde se guardaran las predicciones
    for n_entry in range(N_entry):
        kindex_temp=list(kindex[n_entry])#se crea una lista con los indices de los k vecinos más cercanos
        tag=[]
        for index in kindex_temp:
            tag.append(targets_train[int(index)])#se crea un vector con las clases de los k vecinos mas cercanos
        prediction[n_entry]=max(tag, key = tag.count)
    return prediction

def classification(features_train, features_entry, targets_train,k):
    """
    Realiza toda la tarea de clasificacion del algoritmo a través de las funciones antes descritas

    Parameters
    ----------
    k : int
        Vecinos más cercanos al vector de características a clasificar
    features_train : array
        Matriz con vectores de características de entrenamiento
    features_entry : array
        Matriz con vectores de características a clasificar
    targets_train : array
        vector que contiene las clases dentro del conjunto de entrenamiento

    Returns
    -------
    prediction : array
        Vector con las clases predichas
    """
    C,OmegaC=indexClass(targets_train)
    d,N_entry=distance(features_train, features_entry)
    kindex,kdistance=distanceMin(d,k,N_entry)
    prediction=target(kindex, kdistance, targets_train,N_entry,k)
    return prediction

def fit(k_max,features_train, features_entry, targets_train, targets_entry):
    """
    Se realiza una iteración de evaluaciones con distintos valores de k
    
    Parameters
    ----------
    k_max : int
        Maxima k en la iteración.
    features_train : array
        Matriz con vectores de características de entrenamiento
    features_entry : array
        Matriz con vectores de características a clasificar
    targets_train : TYPE
        DESCRIPTION.
    targets_train : array
        vector que contiene las clases dentro del conjunto de entrenamiento
    """
    for k in range(k_max):
        prediction=classification(features_train, features_entry, targets_train,k+1)
        exactitud=accuracy(prediction, targets_entry)
        print("\nk= "+ str(k+1))
        print("\nExactitud: " + str(exactitud) + "%")

def accuracy(prediction, targets):
    """
    Calcula la exactitud de la clasificación con un conjunto de prueba

    Parameters
    ----------
    prediction : array
        Vector con las clases predichas
    targets : array
        Vector con las clases verdaderas

    Returns
    -------
    float
        Exactitud de 0 a 1

    """
    return (prediction == targets).mean()