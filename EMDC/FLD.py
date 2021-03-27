# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:36:09 2020
@author: Samuel Osorio Gutiérrez
"""
import numpy as np
from matplotlib import pyplot as plt

def indexTarget(targets_train):
    """
    Identification of each target 

    Parameters
    ----------
    targets_train : array
        Array with targets vectors (training)

    Returns
    -------
    C : int
        Number of targets
    OmegaC : list of list
        List of list with index of vectors with same target
    """
    C=max(targets_train) 
    N=targets_train.shape[0]
    OmegaC=[]
    for c in range(C+1): 
        listC=[]
        for n in range (N):
                if targets_train[n] == c:
                    listC.append(n)
        OmegaC.append(listC)  
    return C,OmegaC 

def average(features_train,C,OmegaC):
    """
    Mean of features vectors

    Parameters
    ----------
    features_train : array
        Array with features vectors (training)
    C : int
        Number of targets
    OmegaC : List of list
        List of list with index of vectors with same target

    Returns
    -------
    mc : array
        Means of features vectors with same target
    m : array
        Mean of all features vectors
    D : int
        Dimension of features vectors
    """
    D=features_train.shape[1] 
    mc=np.zeros([C+1,D])
    m=np.zeros(D,)
    for c in range(C+1):
        avg=np.zeros(D,)
        for n in range (len(OmegaC[c])):
            avg+=features_train[OmegaC[c][n],:]
        avg/=len(OmegaC[c])
        mc[c,:]=avg 
        m+=avg
    m/=(C+1)
    return mc,m,D

def covariance (features_train,C,OmegaC,D,mc,m):
    """
    Within-class and between-class ovariance matrix 

    Parameters
    ----------
    features_train : array
        Array with features vectors (training)
    C : int
        Number of targets
    OmegaC : List of list
        List of list with index of vectors with same target
    D : int
        Dimension of features vectors
    mc : array
        Means of features vectors with same target
    m : array
        Mean of all features vectors

    Returns
    -------
    Sw : array
        within-class covariance matrix 
    Sb : array
         between-class ovariance matrix 
    """
    Sw=np.zeros([D,D]) 
    for c in range(C+1):
        Sc=np.zeros([D,D])
        for index in range (len(OmegaC[c])):
            x=features_train[OmegaC[c][index],:]
            a = (x-mc[c,:])
            a = a.reshape((-1, 1))
            Sc+=np.matmul(a,np.transpose(a))
        Sw+=Sc
    Sb=np.zeros([D,D])
    for c in range(C+1):
        a = (mc[c,:]-m)
        a = a.reshape((-1, 1))
        Sb+=len(OmegaC[c])*np.matmul(a,np.transpose(a))    
    return Sw,Sb

def pseudoinverseLDA(Sw,Sb,D):
    """
    Pseudoinverse matrix of Sw and Sb

    Parameters
    ----------
    Sw : array
        within-class covariance matrix 
    Sb : array
         between-class ovariance matrix 
    D : int
        Dimension of features vectors

    Returns
    -------
    Swpseudo : array
        Sw pseudoinverse matrix
    Sbpseudo : array
        Sb pseudoinverse matrix
    """
    uSw, sSw, vhSw=np.linalg.svd(Sw)
    uSb, sSb, vhSb=np.linalg.svd(Sb)
    idensSw=np.zeros([D,D])
    idensSb=np.zeros([D,D])
    for i in range(D):
        idensSw[i][i]=sSw[i]
        idensSb[i][i]=sSb[i]
    invsSw=np.linalg.inv(idensSw)
    invsSb=np.linalg.inv(idensSw)
    Swpseudo=np.matmul(np.matmul(uSw,invsSw),vhSw)
    Sbpseudo=np.matmul(np.matmul(uSb,invsSb),vhSb)
    return Swpseudo,Sbpseudo

    
def eigen(Sw,Sb,D,pLDA):
    """
    Eigen values and eigen vectors
    Parameters
    ----------
    Sw : array
        within-class covariance matrix 
    Sb : array
         between-class ovariance matrix 
    D : int
        Dimension of features vectors
    pLDA: bool
        Boolean that activates pseudoinverse function

    Returns
    -------
    val_p : array
        Eigen values
    vect_p : array
        Eigen vectors
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
    Reduction matrix W

    Parameters
    ----------
    val_p : array
        Eigen values
    vect_p : array
        Eigen vectors
    K : int
        New dimension of features vectors K<=min(C,D)-1
    D : int
        Dimension of features vectors

    Returns
    -------
    W : array
        Reduction Matrix
    """
    listIndex=[] 
    val_p=list(val_p) 
    val_p_aux=list(val_p) 
    W=np.zeros([D,K]) 
    for i in range(K):
        maxValue=max(val_p,key=abs)
        index=val_p.index(maxValue)
        val_p.pop(index)
        index=val_p_aux.index(maxValue)
        listIndex.append(index)
        W[:,i]=vect_p[:,index]   
    return W

def train(features_train,targets_train, K, pLDA=False, plot=False, Terminal=False):
    """
    Train of FLD

    Parameters
    ----------
    features_train : array
        Array with features vectors (training)
    targets_train : array
        Array with targets vectors (training)  
    K : int
        New dimension of features vectors K<=min(C,D)-1
    pLDA : bool, optional
        Boolean that activates pseudoinverse function. The default is False.
    plot : bool, optional
        Indica si se graficara el resultado 2D. The default is False.
    Terminal : bool, optional
        Enable algorithm information. The default is False.

    Returns
    -------
    W : array
        Reduction Matrix
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
    Graph of new features space
        Parameters
    ----------
    features_train : array
        Array with features vectors (training)
    OmegaC : List of list
        List of list with index of vectors with same target
    C : int
        Number of targets
    new_features_train : array
        New features space
    """
    fig, axs=plt.subplots(1,2,False,False)
    for c in range(C+1):
        if new_features_train.shape[1]==1:
            axs[0].scatter(features_train[OmegaC[c],0],features_train[OmegaC[c],1])   
            axs[1].scatter(new_features_train[OmegaC[c],0],np.zeros(len(OmegaC[c],))) 
        else:
            axs[0].scatter(features_train[OmegaC[c],0],features_train[OmegaC[c],1])   
            axs[1].scatter(new_features_train[OmegaC[c],0],new_features_train[OmegaC[c],1])