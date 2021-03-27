# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:37:49 2020

@author: Samuel Osorio Gutiérrez
"""
import numpy as np

def indexTargets(targets_train):
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
    OmegaK : list of list
        List of list with index of vectors with same target
    """
    K=max(targets_train)
    N=targets_train.shape[0]
    OmegaK=[]
    for c in range(K+1): 
        listK=[]
        for n in range (N):
                if targets_train[n] == c:
                    listK.append(n)
        OmegaK.append(listK)
    return K,N, OmegaK

def Apriori(features_train, K, OmegaK):
    """
    A priori probabilities

    Parameters
    ----------
    features_train : array
        Array with features vectors (training)
    K : int
        Number of targets
    OmegaK : List of list
        List of list with index of vectors with same target

    Returns
    -------
    pi_k : array
        Vector with a priori probabilities of each target
    """
    N=features_train.shape[0]
    pi_k=np.zeros(K+1)
    for k in range(K+1):
        n_k=len(OmegaK[k])
        pi_k[k]=n_k/N
    pi_k= pi_k.reshape((-1, 1))
    return pi_k

def average(features_train,K,OmegaK):
    """
    Mean of features vectors

    Parameters
    ----------
    features_train : array
        Array with features vectors (training)
    K : int
        Number of targets
    OmegaK : List of list
        List of list with index of vectors with same target

    Returns
    -------
    m_k : array
        Means of features vectors with same target
    D : int
        Dimension of features vectors
    """
    D=features_train.shape[1]
    m_k=np.zeros([K+1,D])
    for k in range(K+1):
        avg=np.zeros(D,)
        for n in range (len(OmegaK[k])):
            x=features_train[OmegaK[k][n],:]
            avg+=x 
        avg/=len(OmegaK[k])
        m_k[k,:]=avg 
    return m_k,D

def pseudoinverseLDA(S,D):
    """
    Pseudoinverse matrix generation

    Parameters
    ----------
    S : array
        Covariance matrix
    D : int
        Dimension of features vectors

    Returns
    -------
    Spseudo : array
        Pseudoinverse matrix of S

    """
    uS, sS, vhS=np.linalg.svd(S)
    idensS=np.zeros([D,D])
    for i in range(D):
        idensS[i][i]=sS[i]
    invsS=np.linalg.inv(idensS)
    Spseudo=np.matmul(np.matmul(uS,invsS),vhS)
    return Spseudo

def covariance (features_train,K,OmegaK,D,m_k, N, pLDA):
    """
    Covariance matrix

    Parameters
    ----------
    features_train : array
        Array with features vectors (training)
    K : int
        Number of targets
    OmegaK : List of list
        List of list with index of vectors with same target
    D : int
        Dimension of features vectors
    m_k : array
        Means of features vectors with same target
    pLDA : bool
        Boolean that activates pseudoinverse function.

    Returns
    -------
    S_inv : array
        Inverse covariance matrix 
    """
    S=np.zeros([D,D]) 
    for k in range(K+1):
        Sk=np.zeros([D,D])
        for index in range (len(OmegaK[k])):
            x=features_train[OmegaK[k][index],:]
            a = (x-m_k[k,:])
            a = a.reshape((-1, 1))
            Sk+=np.matmul(a,np.transpose(a))
        S+=Sk 
    S/=(N-K)
    if pLDA:
        S_inv=pseudoinverseLDA(S,D)
    else:
        S_inv=np.linalg.inv(S)
    return S_inv


def classification(S_inv,m_k,pi_k,features_entry):
    """
    Classification of features vectors

    Parameters
    ----------
    S_inv : array
        Inverse covariance matrix 
    m_k : array
        Means of features vectors with same target
    pi_k : array
        Vector with a priori probabilities of each target
    features_entry : array
        Array with features vectors

    Returns
    -------
    prediction : array
        Arrar with predicted targets
    """
    K=pi_k.shape[0]
    N=features_entry.shape[0]
    prediction=np.zeros(N)
    for n in range (N):   
        delta=[]
        for k in range(K):
            a=features_entry[n,:]-m_k[k,:]
            delta_temp=(-1/2)*np.matmul(np.matmul(a,S_inv),np.transpose(a))+np.log(pi_k[k,:])
            delta.append(delta_temp)
        maxValue=max(delta)
        prediction[n]=delta.index(maxValue)
    return prediction

def train(features_train,targets_train, pLDA=False, Terminal=False):
    """
    Train of LDA

    Parameters
    ----------
    features_train : array
        Array with features vectors (training)
    targets_train : array
        Array with targets vectors (training)
    pLDA : bool, optional
        Boolean that activates pseudoinverse function. The default is False.
    Terminal : bool, optional
        Enable algorithm information. The default is False.

    Returns
    -------
    S_inv : array
        Inverse covariance matrix 
    m_k : array
        Means of features vectors with same target
    pi_k : array
        Vector with a priori probabilities of each target
    """
    K,N,OmegaK=indexTargets(targets_train)
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
    Accuracy of test set

    Parameters
    ----------
    prediction : array
        Predicted targets vectors 
    targets : array
        Targets vectors of test set

    Returns
    -------
    float
        Acurracy [0,1]
    """
    return (prediction == targets).mean()
    
def LOOCV(features_train,targets_train,pLDA):
    """
    Train of LDA

    Parameters
    ----------
    features_train : array
        Array with features vectors (training)
    targets_train : array
        Array with targets vectors (training)
    pLDA : bool, optional
        Boolean that activates pseudoinverse function. The default is False.
    Returns
    -------
    acc: float
        Acurracy [0,1]
    """
    samples=features_train.shape[0]
    acc=0
    for i in range(samples):
        temp_features_train=np.delete(features_train,i,axis=0)
        temp_targets_train=np.delete(targets_train,i,axis=0)
        sample_feature=features_train[i,:]
        sample_target=targets_train[i]
        sample_feature=sample_feature.reshape((1,-1))
        S_inv,m_k,pi_k=train(temp_features_train,temp_targets_train, pLDA,Terminal=False)
        prediction=classification(S_inv,m_k,pi_k, sample_feature)
        acc+=accuracy(prediction,sample_target)
    acc/=samples
    return acc