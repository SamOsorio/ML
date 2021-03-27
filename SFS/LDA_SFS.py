# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:37:49 2020

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
    K : int
        indica el numero máximo de clases dentro del conjunto de entrenamiento
    OmegaK : list of list
        lista de listas que contiene los indices correspondientes a cada clase del conjunto de entrenamiento
    """
    K=max(targets_train) #se obtiene el maximo numero de clases
    N=targets_train.shape[0]
    OmegaK=[]
    for c in range(K+1): #se busca cada clase dentro del arreglo para agregar sus indices a una lista
        listK=[]
        for n in range (N):
                if targets_train[n] == c:
                    listK.append(n)
        OmegaK.append(listK)  #se crea una lista con las listas de indices de cada clase
    return K,N, OmegaK

def Apriori(features_train, K, OmegaK):
    """
    Devuelve la probabilidad a priori

    Parameters
    ----------
    features_train : array
        Matriz con vectores de características de entrenamiento
    K : int
        Número de clases
    OmegaK : list of list
        lista de listas con los indices de los vectores de cada clase

    Returns
    -------
    pi_k : array
        Vector con la probabilidad a priori para cada clase
    """
    N=features_train.shape[0]
    pi_k=np.zeros(K+1)
    for k in range(K+1):
        n_k=len(OmegaK[k])
        pi_k[k]=n_k/N
    return pi_k

def average(features_train,K,OmegaK):
    """
    Realiza el promedio de todos los vectores en cada clase, así como el promedio total

    Parameters
    ----------
    features_train : array
        Matriz que contiene los vectores de características de entrenamiento
    K : int
        Número total de clases
    OmegaK : List of list
        Contiene los indices de los vectores correspondientes a cada clase

    Returns
    -------
    mk : array
        Matriz que contiene las medias de todos los vectores de características en cada clase

    D : int
        Dimensiones de los vectores de características
    """
    D=features_train.shape[1] #dimension de los vectores de caracteristicas
    mk=np.zeros([K+1,D]) #se crea una matriz de ceros que contiene el promedio de las caracteristicas en cada clase
    for k in range(K+1):#se saca el promedio de cada clase
        avg=np.zeros(D,)
        for n in range (len(OmegaK[k])): #se hace un corrimiento por los indices de dicha clase
            x=features_train[OmegaK[k][n],:]
            avg+=x #se suman los vectores que tengan dichos indices
        avg/=len(OmegaK[k])
        mk[k,:]=avg #se guarda en la matriz el promedio de la clase
    return mk,D

def pseudoinverseLDA(S,D):
    """
    Se obtiene la matriz pseudoinversa en caso de que no exista la inversa

    Parameters
    ----------
    S : array
        Matriz de covarianza 
    D : int
        Dimensiones de los vectores de características

    Returns
    -------
    Spseudo : array
        Matriz pseudoinversa de S

    """
    uS, sS, vhS=np.linalg.svd(S)
    idensS=np.zeros([D,D])
    for i in range(D):#se genera una matriz identidad
        idensS[i][i]=sS[i]
    invsS=np.linalg.inv(idensS)
    Spseudo=np.matmul(np.matmul(uS,invsS),vhS)
    return Spseudo

def covariance (features_train,K,OmegaK,D,m_k, N, pLDA):
    """
    Realiza la matriz de covarianza promedio

    Parameters
    ----------
    features_train : array
        Matriz que contiene los vectores de características de entrenamiento
    K : int
        Número total de clases
    OmegaK : List of list
        Contiene los indices de los vectores correspondientes a cada clase
    D : int
        Dimensiones de los vectores de características
    m_k : array
        Matriz que contiene las medias de todos los vectores de características en cada clase
    pLDA : bool
        Indica si se hará la matriz inversa o pseudoinversa

    Returns
    -------
    S_inv : array
        Matriz de covarianza promedio inversa
    """
    S=np.zeros([D,D]) #matriz de covarianza promedio
    for k in range(K+1):#se saca la covarianza de cada clase
        Sk=np.zeros([D,D])
        for index in range (len(OmegaK[k])): #se hace un corrimiento por los indices de dicha clase
            x=features_train[OmegaK[k][index],:]
            a = (x-m_k[k,:])
            a = a.reshape((-1, 1))
            Sk+=np.matmul(a,np.transpose(a))
        S+=Sk #se guarda la matriz de covarianza de la respectiva clase
    S/=(N-K)
    if pLDA:
        S_inv=pseudoinverseLDA(S,D)
    else:
        S_inv=np.linalg.inv(S)
    return S_inv


def classification(S_inv,m_k,pi_k,features_entry):
    """
    Se realiza la clasificación con los parámetros estimados en el entrenamiento

    Parameters
    ----------
    S_inv : array
        Matriz de covarianza inversa
    m_k : array
        Matriz con los vectores de medias en cada clase
    pi_k : array
        Vector con los valores de la probabilidad a priori en cada clase
    features_entry : array
        Matriz con vectores de características a clasificar

    Returns
    -------
    prediction : array
        Vector con las clases estimadas
    """
    K=pi_k.shape[0]
    N=features_entry.shape[0]
    prediction=np.zeros(N)
    for n in range (N):   
        delta=[]
        for k in range(K):
            a=features_entry[n,:]-m_k[k,:]
            delta_temp=(-1/2)*np.matmul(np.matmul(a,S_inv),np.transpose(a))+np.log(pi_k[k])
            delta.append(delta_temp)
        maxValue=max(delta)
        prediction[n]=delta.index(maxValue)
    return prediction

def sphere(S):
    """
    Realizar una operación para que la matriz de covarianza sea una matriz identidad (no funciona si S tiene valores negativos)

    Parameters
    ----------
    S : array
        Matriz de covarianza

    Returns
    -------
    idenU : array
        Matriz identidad con los valores propios
    D : array
        Matriz con los vectores propios

    """
    U,D=np.linalg.eig(S)
    dim=U.shape[0]
    idenU=np.zeros([dim,dim])
    for i in range(dim):#se genera una matriz identidad
        idenU[i][i]=U[i]
    print(D)
    D=np.sqrt(D)
    print(D)
    return idenU,D

def train(features_train,targets_train, pLDA=False, Terminal=False):
    """
    Realizar el entrenamiento del algoritmos

    Parameters
    ----------
    features_train : array
        Matriz con vectores de entrenamiento
    targets_train : array
        Vector de clases
    pLDA : bool, optional
        Indica si harás una matriz inversa o pseudoinversa durante el entrenamiento. The default is False.
    Terminal : bool, optional
        Indica si quieres mostrar información en la terminal. The default is False.

    Returns
    -------
    S_inv : array
        Matriz de covarianza inversa.
    m_k : array
        Matriz con las medias de los vectores en cada clase
    pi_k : array
        Vector con las probabilidades a priori
    """
    K,N,OmegaK=indexClass(targets_train)
    pi_k=Apriori(features_train, K, OmegaK)
    m_k,D=average(features_train,K,OmegaK)
    S_inv=covariance (features_train,K,OmegaK,D,m_k, N, pLDA)
    if Terminal:
        print("\nS_inv:")
        print(S_inv)
        print("\nm_k:")
        print(m_k)
        print("\npi_k:")
        print(pi_k)
    return S_inv,m_k,pi_k

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
    
