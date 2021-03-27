# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:38:08 2020

@author: Samuel Osorio Gutiérrez
"""
import numpy as np

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

def normalization(features_entry):
   D=features_entry.shape[1]
   N=features_entry.shape[0]
   m_t=np.zeros(D)
   s_t=np.zeros(D)
   for d in range(D):
       for n in range(N):
           m_t[d]+=features_entry[n,d]
   m_t/=N
   for d in range(D):
       s_t[d]=np.matmul(features_entry[:,d]-m_t[d],features_entry[:,d]-m_t[d])
   s_t=np.sqrt(s_t/(N-1))
   
   for d in range(D):
       for n in range(N):
           features_entry[n,d]=(features_entry[n,d]-m_t[d])/s_t[d]
   return features_entry