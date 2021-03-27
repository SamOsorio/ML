# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:36:09 2020
@author: Samuel Osorio Gutiérrez
"""
import numpy as np
from matplotlib import pyplot as plt

def indexTarget(targets_train):
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
    C=max(targets_train) #se obtiene el máximo número de clases
    N=targets_train.shape[0]
    OmegaC=[]
    for c in range(C+1): #se busca cada clase dentro del arreglo para agregar sus indices a una lista
        listC=[]
        for n in range (N):
                if targets_train[n] == c:
                    listC.append(n)
        OmegaC.append(listC)  #se crea una lista con las listas de indices de cada clase
    return C,OmegaC 

def average(features_train,C,OmegaC):
    """
    Realiza el promedio de todos los vectores en cada clase, así como el promedio total

    Parameters
    ----------
    features_train : array
        Matriz que contiene los vectores de características de entrenamiento
    C : int
        Número total de clases
    OmegaC : List of list
        Contiene los indices de los vectores correspondientes a cada clase

    Returns
    -------
    mc : array
        Matriz que contiene las medias de todos los vectores de características en cada clase
    m : array
        Vector que contiene la media total de todos los vectores de caacteristicas
    D : int
        Dimensiones de los vectores de características
    """
    D=features_train.shape[1] #dimension de los vectores de caracteristicas
    mc=np.zeros([C+1,D]) #se crea una matriz de ceros que contiene el promedio de las caracteristicas en cada clase
    m=np.zeros(D,)#vector que contendrá el promedio total de las caracteristicas
    for c in range(C+1):#se saca el promedio de cada clase
        avg=np.zeros(D,)
        for n in range (len(OmegaC[c])): #se hace un corrimiento por los indices de dicha clase
            avg+=features_train[OmegaC[c][n],:]#se suman los vectores que tengan dichos indices
        avg/=len(OmegaC[c])
        mc[c,:]=avg #se guarda en la matriz el promedio de la clase
        m+=avg
    m/=(C+1)#promedio total
    return mc,m,D

def covariance (features_train,C,OmegaC,D,mc,m):
    """
    Realiza la matriz de covarianza entre calses y dentro de cada clase

    Parameters
    ----------
    features_train : array
        Matriz que contiene los vectores de características de entrenamiento
    C : int
        Número total de clases
    OmegaC : List of list
        Contiene los indices de los vectores correspondientes a cada clase
    D : int
        Dimensiones de los vectores de características
    mc : array
        Matriz que contiene las medias de todos los vectores de características en cada clase
    m : array
        Vector que contiene la media total de todos los vectores de caacteristicas

    Returns
    -------
    Sw : array
        Matriz de covarianza dentro de cada clase
    Sb : array
        Matriz de covarianza entre clases
    """
    Sw=np.zeros([D,D]) #matriz de covarianza dentro de cada clase
    for c in range(C+1):#se saca la covarianza de cada clase
        Sc=np.zeros([D,D])
        for index in range (len(OmegaC[c])): #se hace un corrimiento por los indices de dicha clase
            x=features_train[OmegaC[c][index],:]
            a = (x-mc[c,:])
            a = a.reshape((-1, 1))
            Sc+=np.matmul(a,np.transpose(a))
        Sw+=Sc #se suman todas las matrices de covarianza dentro de cada clase
    Sb=np.zeros([D,D])#matriz de covarianza entre clases
    for c in range(C+1):#se saca la covarianza entre clases
        a = (mc[c,:]-m)
        a = a.reshape((-1, 1))
        Sb+=len(OmegaC[c])*np.matmul(a,np.transpose(a))    
    return Sw,Sb

def pseudoinverseLDA(Sw,Sb,D):
    """
    Se obtiene la matriz pseudoinversa en caso de que no exista la inversa

    Parameters
    ----------
    Sw : array
        Matriz de covarianza dentro de cada clase
    Sb : array
        Matriz de covarianza entre clases
    D : int
        Dimensiones de los vectores de características

    Returns
    -------
    Swpseudo : array
        Matriz pseudoinversa de Sw
    Sbpseudo : array
        Matriz pseudoinversa de Sb
    """
    uSw, sSw, vhSw=np.linalg.svd(Sw)
    uSb, sSb, vhSb=np.linalg.svd(Sb)
    idensSw=np.zeros([D,D])
    idensSb=np.zeros([D,D])
    for i in range(D):#se genera una matriz identidad
        idensSw[i][i]=sSw[i]
        idensSb[i][i]=sSb[i]
    invsSw=np.linalg.inv(idensSw)
    invsSb=np.linalg.inv(idensSw)
    Swpseudo=np.matmul(np.matmul(uSw,invsSw),vhSw)
    Sbpseudo=np.matmul(np.matmul(uSb,invsSb),vhSb)
    return Swpseudo,Sbpseudo

    
def eigen(Sw,Sb,D,pLDA):
    """
    se obtienen los valores y vectores propios para generar un nuevo espacio
    Parameters
    ----------
    Sw : array
        Matriz de covarianza dentro de cada clase
    Sb : array
        Matriz de covarianza entre clases
    D : int
        Dimensiones de los vectores de características
    pLDA : bool
        Indica si se hará una matriz inversa o pseudoinversa

    Returns
    -------
    val_p : array
        Vector de valores propios
    vect_p : array
        Matriz con los vectores propios  
    """
    if pLDA:
        Swpseudo,Sbpseudo=pseudoinverseLDA(Sw,Sb,D)
        A=np.matmul(Swpseudo,Sb)
    else:
        A=np.matmul(np.linalg.inv(Sw),Sb)
    val_p,vect_p=np.linalg.eig(A)
    return val_p,vect_p


def reductionMatrix(val_p,vect_p,K,D):
    """
    Generación de la matriz de reducción dimensional W

    Parameters
    ----------
    val_p : array
        Vector de valores propios
    vect_p : array
        Matriz con los vectores propios
    K : int
        Dimensión a la que se quiere reducir K<=min(C,D)-1
    D : int
        Dimensiones de los vectores de características

    Returns
    -------
    W : array
        Matriz de reducción
    """
    listIndex=[] #se crea una lista con los indices de los valores propios mas grandes
    val_p=list(val_p) #se convierte el vector en lista
    val_p_aux=list(val_p) #se convierte el vector en lista
    W=np.zeros([D,K]) #se crea la matriz de transformación para la reducción del espacio
    for i in range(K):#se busca el valor máximo, se guarda su indice, se elimina de la lista y se guarda el vector correspondiente 
        maxValue=max(val_p,key=abs)
        index=val_p.index(maxValue)
        val_p.pop(index)
        index=val_p_aux.index(maxValue)
        listIndex.append(index)
        W[:,i]=vect_p[:,index]    
    return W

def train(features_train,targets_train,K, pLDA=False, plot=False, Terminal=False):
    """
    Se realiza paso a paso el algoritmo con el objetivo de obtener la matriz de transformacion

    Parameters
    ----------
    features_train : array
        Matriz que contiene los vectores de características de entrenamiento
    targets_train : array
        vector que contiene las clases dentro del conjunto de entrenamiento
    K : int
        Dimensión a la que se quiere reducir K<=min(C,D)-1
    pLDA : bool, optional
        Indica si harás una matriz inversa o pseudoinversa durante el entrenamiento. The default is False.
    plot : bool, optional
        Indica si se graficara el resultado 2D. The default is False.
    Terminal : bool, optional
        Indica si quieres mostrar información en la terminal. The default is False.

    Returns
    -------
    W : array
        Matriz de reducción
    """
    C,OmegaC=indexTarget(targets_train)
    mc,m,D=average(features_train,C,OmegaC)
    Sw,Sb= covariance(features_train,C,OmegaC,D,mc,m)
    if mc.shape[0]==2:
        W=mc[1,:]-mc[0,:]
        val_p=0
        vect_p=0
        if pLDA:
            Swpseudo,Sbpseudo=pseudoinverseLDA(Sw,Sb,D)
            W=np.matmul(Swpseudo,W)
        else:
            W=np.matmul(np.linalg.inv(Sw),W)
        W=W.reshape((-1, 1))
    else:
        val_p,vect_p=eigen(Sw,Sb,D,pLDA)
        W=reductionMatrix(val_p,vect_p,K,D)
    
    if Terminal:
        print("Eigen values: ")
        print(val_p)
        print("\nEigen vectors: ")
        print(vect_p)
        print("\nReduction Matrix W: ")
        print(W)
        
    if plot:
        new_features_train=np.matmul(features_train,W)
        plot2D(features_train,new_features_train,C,OmegaC)
    return W

def plot2D(features_train,new_features_train,C,OmegaC):
    """
    Recibe el espacio original y el espacio nuevo y los grafica
        Parameters
    ----------
    features_train : array
        Matriz que contiene los vectores de características de entrenamiento
    OmegaC : list of list
        lista de listas que contiene los indices correspondientes a cada clase del conjunto de entrenamiento
    C : int
        indica el numero máximo de clases dentro del conjunto de entrenamiento
    new_features_train : array
        Matriz de caracteristicas reducidas
    """
    fig, axs=plt.subplots(1,2,False,False)
    for c in range(C+1):
        if new_features_train.shape[1]==1:
            axs[0].scatter(features_train[OmegaC[c],0],features_train[OmegaC[c],1])   
            axs[1].scatter(new_features_train[OmegaC[c],0],np.zeros(len(OmegaC[c],))) 
        else:
            axs[0].scatter(features_train[OmegaC[c],0],features_train[OmegaC[c],1])   
            axs[1].scatter(new_features_train[OmegaC[c],0],new_features_train[OmegaC[c],1])