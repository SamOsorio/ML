# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:54:25 2020

@author: Samuel Osorio Gutiérrez
"""
# import kNN_SFS #test
# import LDA_SFS
from SFS import kNN_SFS,LDA_SFS, MLP_SFS
import numpy as np

def SFS_wrapper(features_train, features_test, targets_train, targets_test, K, **algorithm):
    D=features_train.shape[1] #dimension de los vectores de caracteristicas
    new_features_train=[]#lista con los nuevos vectores reducidos para el entrenamiento
    new_features_test=[]#lista con los nuevos vectores reducidos para el test
    featuresIndex=[*range(D)] #se crea una lista con los indices de cada caracteristica
    for k in range(K): #se realiza el proceso hasta tener las k caracteristicas deseadas
        accs=[]#lista con exactitudes
        index_d=[]#lista con los indices utilizados en el ciclo
        for d in featuresIndex:  #para cada característica
            new_features_train.append(features_train[:,d])#se agrega una caracteristica a los conjuntos
            new_features_test.append(features_test[:,d])
            new_features_train=np.array(new_features_train)
            new_features_test=np.array(new_features_test)
            if k==0:#hay una consideración especial para el incio pues se necesita una matriz de nx1 en lugar de un vector de n
                new_features_train = new_features_train.reshape((-1, 1))
                new_features_test = new_features_test.reshape((-1, 1))
            else:
                new_features_train=np.transpose(new_features_train)# se realiza transpuesta porque la lista los ordena al revés
                new_features_test=np.transpose(new_features_test)
            if algorithm['name'] =="kNN":                   
                prediction=kNN_SFS.classification(new_features_train, new_features_test, targets_train,algorithm['k'])
                acc=kNN_SFS.accuracy(prediction,targets_test)
            elif algorithm['name'] =="LDA":
                S_inv,m_k,pi_k=LDA_SFS.train(new_features_train,targets_train, algorithm['pLDA'],Terminal=False)
                prediction=LDA_SFS.classification(S_inv,m_k,pi_k, new_features_test)
                acc=LDA_SFS.accuracy(prediction,targets_test)
            elif algorithm['name']=="MLP":
                theta= MLP_SFS.train( new_features_train,targets_train,algorithm['hidden_layers'],algorithm['units'], algorithm['iterations'],algorithm['rate'], activation_function=algorithm['activation_function'], Type="stochastic",Regularization=False, lamb=0.00001)
                prediction=MLP_SFS.classification(new_features_test,theta,activation_function=algorithm['activation_function'])
                MLP_targets_test=MLP_SFS.targets_conv(targets_test,algorithm['activation_function'])
                acc=MLP_SFS.accuracy(MLP_targets_test,prediction)
            accs.append(acc)#se almacenan todas las exactitudes obtenidas
            index_d.append(d)#se agrega el indice con el que se trabajo en el cilo
            new_features_train=np.transpose(new_features_train) #se realiza transpuesta para volver lista
            new_features_test=np.transpose(new_features_test)
            new_features_train=list(new_features_train)
            new_features_train.pop(k)#se elimina el vector de la lista para probar con otro
            new_features_test=list(new_features_test)
            new_features_test.pop(k)
        maxValue=max(accs)#se identifica la exactitud con maximo valor
        index=accs.index(maxValue)#se identifica cual fue el vector con la mayor exactitud
        index=index_d[index]#se utiliza actualiza el indice tomando en cosnideracion los eliminados
        new_features_train.append(features_train[:,index])#se agrega la característica con mejor desempeño
        new_features_test.append(features_test[:,index])
        featuresIndex.remove(index)#se elimina la característica agregada de los vectores que se analizaran posteriormente
#        print(accs)
#        print(featuresIndex)
    new_features_train=np.array(new_features_train)
    new_features_test=np.array(new_features_test)
    new_features_train=np.transpose(new_features_train)
    new_features_test=np.transpose(new_features_test)
    return new_features_train, new_features_test

def SFFS_wrapper(features_train, features_test, targets_train, targets_test, K, **algorithm):
    D=features_train.shape[1] #dimension de los vectores de caracteristicas
    new_features_train=[]#lista con los nuevos vectores reducidos para el entrenamiento
    new_features_test=[]#lista con los nuevos vectores reducidos para el test
    featuresIndex=[*range(D)] #se crea una lista con los indices de cada caracteristica
    IndexAdd=[] #lista con indices agregados
    k=0
    D_new=0
    while k<K: #se realiza el proceso hasta tener las k caracteristicas deseadas
        accs=[]#lista con exactitudes
        index_d=[]#lista con los indices utilizados en el ciclo
        for d in featuresIndex:  #para cada característica
            new_features_train.append(features_train[:,d])#se agrega una caracteristica a los conjuntos
            new_features_test.append(features_test[:,d])
            new_features_train=np.array(new_features_train)
            new_features_test=np.array(new_features_test)
            if k==0:#hay una consideración especial para el incio pues se necesita una matriz de nx1 en lugar de un vector de n
                new_features_train = new_features_train.reshape((-1, 1))
                new_features_test = new_features_test.reshape((-1, 1))
            else:
                new_features_train=np.transpose(new_features_train)# se realiza transpuesta porque la lista los ordena al revés
                new_features_test=np.transpose(new_features_test)
            if algorithm['name'] =="kNN":                   
                prediction=kNN_SFS.classification(new_features_train, new_features_test, targets_train,algorithm['k'])
                acc=kNN_SFS.accuracy(prediction,targets_test)
            elif algorithm['name'] =="LDA":
                S_inv,m_k,pi_k=LDA_SFS.train(new_features_train,targets_train, algorithm['pLDA'],Terminal=False)
                prediction=LDA_SFS.classification(S_inv,m_k,pi_k, new_features_test)
                acc=LDA_SFS.accuracy(prediction,targets_test)
            elif algorithm['name']=="MLP":
                theta= MLP_SFS.train( new_features_train,targets_train,algorithm['hidden_layers'],algorithm['units'], algorithm['iterations'],algorithm['rate'], activation_function=algorithm['activation_function'], Type="stochastic",Regularization=False, lamb=0.00001)
                prediction=MLP_SFS.classification(new_features_test,theta,activation_function=algorithm['activation_function'])
                MLP_targets_test=MLP_SFS.targets_conv(targets_test,algorithm['activation_function'])
                acc=MLP_SFS.accuracy(MLP_targets_test,prediction)
            accs.append(acc)#se almacenan todas las exactitudes obtenidas
            index_d.append(d)#se agrega el indice con el que se trabajo en el ciclo
            new_features_train=np.transpose(new_features_train) #se realiza transpuesta para volver lista
            new_features_test=np.transpose(new_features_test)
            new_features_train=list(new_features_train)
            new_features_train.pop(k)#se elimina el vector de la lista para probar con otro
            new_features_test=list(new_features_test)
            new_features_test.pop(k)           
        maxValue=max(accs)#se identifica la exactitud con maximo valor
        index=accs.index(maxValue)#se identifica cual fue el vector con la mayor exactitud
        index=index_d[index]#se utiliza actualiza el indice tomando en cosnideracion los eliminados
        IndexAdd.append(index) #se guarda el indice seleccioando
        new_features_train.append(features_train[:,index])#se agrega la característica con mejor desempeño
        new_features_test.append(features_test[:,index])
        featuresIndex.remove(index)#se elimina la característica agregada de los vectores que se analizaran posteriormente
        k+=1
        while k>2:
            D_new=len(new_features_train)
            new_featuresIndex=[*range(D_new)] #se crea una lista con los indices de cada caracteristica
            accs=[]#lista con exactitudes
            index_d=[]#lista con los indices utilizados en el ciclo
            for d in new_featuresIndex:  #para cada característica agregada
                d_new=IndexAdd[d]
                new_features_train.pop(0)#se quita una caracteristica a los conjuntos
                new_features_test.pop(0)
                new_features_train=np.array(new_features_train)
                new_features_test=np.array(new_features_test)
                new_features_train=np.transpose(new_features_train)# se realiza transpuesta porque la lista los ordena al revés
                new_features_test=np.transpose(new_features_test)
                if algorithm['name'] =="kNN":                   
                    prediction=kNN_SFS.classification(new_features_train, new_features_test, targets_train,algorithm['k'])
                    acc=kNN_SFS.accuracy(prediction,targets_test)
                elif algorithm['name'] =="LDA":
                    S_inv,m_k,pi_k=LDA_SFS.train(new_features_train,targets_train, algorithm['pLDA'],Terminal=False)
                    prediction=LDA_SFS.classification(S_inv,m_k,pi_k, new_features_test)
                    acc=LDA_SFS.accuracy(prediction,targets_test)
                elif algorithm['name']=="MLP":
                    theta= MLP_SFS.train( new_features_train,targets_train,algorithm['hidden_layers'],algorithm['units'], algorithm['iterations'],algorithm['rate'], activation_function=algorithm['activation_function'], Type="stochastic",Regularization=False, lamb=0.00001)
                    prediction=MLP_SFS.classification(new_features_test,theta,activation_function=algorithm['activation_function'])
                    MLP_targets_test=MLP_SFS.targets_conv(targets_test,algorithm['activation_function'])
                    acc=MLP_SFS.accuracy(MLP_targets_test,prediction)  
                accs.append(acc)#se almacenan todas las exactitudes obtenidas
                index_d.append(d_new)#se agrega el indice con el que se trabajo en el ciclo
                new_features_train=np.transpose(new_features_train) #se realiza transpuesta para volver lista
                new_features_test=np.transpose(new_features_test)
                new_features_train=list(new_features_train)
                new_features_train.append(features_train[:,d_new])#se agrega de nuevo caracteristica a los conjuntos
                new_features_test=list(new_features_test)
                new_features_test.append(features_test[:,d_new])#se agrega de nuevo caracteristica a los conjuntos
            maxValueRemove=max(accs)#se identifica la exactitud con maximo valor
            if maxValueRemove<=maxValue:
                break
            maxValue=maxValueRemove
            index_remv=accs.index(maxValueRemove)#se identifica cual fue el vector con la mayor exactitud
            index=index_d[index_remv]#se utiliza actualiza el indice de los ya seleccionados previamente
            featuresIndex.append(index) #se vuelve a agregar el indice seleccionado a la lista original
            new_features_train.pop(index_remv)#se remueve la característica para un mejor desempeño
            new_features_test.pop(index_remv)
            IndexAdd.pop(index_remv)
            k-=1
    new_features_train=np.array(new_features_train)#se convierte en un arreglo y se hace la transpuesta para mantener formato
    new_features_test=np.array(new_features_test)
    new_features_train=np.transpose(new_features_train)
    new_features_test=np.transpose(new_features_test)
    return new_features_train, new_features_test


def SFFS_wrapper_2(features_train, features_test, targets_train, targets_test, pLDA):
    """
    Sequential Forward Floating Selection (wrapper method)

    Parameters
    ----------
    features_train : array
        Array with features vectors (training)
    features_test : array
        Array with features vectors (test)
    targets_train : array
        Array with targets vectors (training)  
    targets_test : array
        Array with targets vectors (test)   
    pLDA: bool
        Boolean that activates pLDA in LDA function
        
    Returns
    -------
    new_features_train: array
        Array with selected features (training)
    new_features_test: array
        Array with selected features (test)
    index_cod: list
        List with coded index of selected features
    """    
    D=features_train.shape[1]
    new_features_train=[]
    new_features_test=[]
    featuresIndex=[*range(D)] 
    IndexAdd=[] 
    k=0
    maxValue=0
    while k<D: 
        accs=[]
        index_d=[]
        for d in featuresIndex:  
            new_features_train.append(features_train[:,d])
            new_features_test.append(features_test[:,d])
            new_features_train=np.array(new_features_train)
            new_features_test=np.array(new_features_test)
            if k==0:
                new_features_train = new_features_train.reshape((-1, 1))
                new_features_test = new_features_test.reshape((-1, 1))
            else:
                new_features_train=np.transpose(new_features_train)
                new_features_test=np.transpose(new_features_test)
            S_inv,m_k,pi_k=LDA.train(new_features_train,targets_train, pLDA,Terminal=False)
            prediction=LDA.classification(S_inv,m_k,pi_k, new_features_test)
            acc=LDA.accuracy(prediction,targets_test)
            accs.append(acc)
            index_d.append(d)
            new_features_train=np.transpose(new_features_train) 
            new_features_test=np.transpose(new_features_test)
            new_features_train=list(new_features_train)
            new_features_train.pop(k)
            new_features_test=list(new_features_test)
            new_features_test.pop(k)           
        max_prev=maxValue
        maxValue=max(accs)
        if maxValue<max_prev:
            break
        index=accs.index(maxValue)
        index=index_d[index]
        IndexAdd.append(index) 
        new_features_train.append(features_train[:,index])
        new_features_test.append(features_test[:,index])
        featuresIndex.remove(index)
        k+=1
        while k>2:
            D_new=len(new_features_train)
            new_featuresIndex=[*range(D_new)] 
            accs=[]
            index_d=[]
            for d in new_featuresIndex: 
                d_new=IndexAdd[d]
                new_features_train.pop(0)
                new_features_test.pop(0)
                new_features_train=np.array(new_features_train)
                new_features_test=np.array(new_features_test)
                new_features_train=np.transpose(new_features_train)
                new_features_test=np.transpose(new_features_test)
                S_inv,m_k,pi_k=LDA.train(new_features_train,targets_train,pLDA,Terminal=False)
                prediction=LDA.classification(S_inv,m_k,pi_k, new_features_test)
                acc=LDA.accuracy(prediction,targets_test) 
                accs.append(acc)
                index_d.append(d_new)
                new_features_train=np.transpose(new_features_train) 
                new_features_test=np.transpose(new_features_test)
                new_features_train=list(new_features_train)
                new_features_train.append(features_train[:,d_new])
                new_features_test=list(new_features_test)
                new_features_test.append(features_test[:,d_new])
            maxValueRemove=max(accs)
            if maxValueRemove<=maxValue:
                break
            maxValue=maxValueRemove
            index_remv=accs.index(maxValueRemove)
            index=index_d[index_remv]
            featuresIndex.append(index) 
            new_features_train.pop(index_remv)
            new_features_test.pop(index_remv)
            IndexAdd.pop(index_remv)
            k-=1
    new_features_train=np.array(new_features_train)
    new_features_test=np.array(new_features_test)
    new_features_train=np.transpose(new_features_train)
    new_features_test=np.transpose(new_features_test)
    
    return new_features_train, new_features_test

def SFFS_wrapper_LOOCV(features_train, targets_train, pLDA):
    """
    Sequential Forward Floating Selection (wrapper method)

    Parameters
    ----------
    features_train : array
        Array with features vectors (training)
    targets_train : array
        Array with targets vectors (training)    
    pLDA: bool
        Boolean that activates pLDA in LDA function
        
    Returns
    -------
    new_features_train: array
        Array with selected features (training)
    index_cod: list
        List with coded index of selected features
    """    
    D=features_train.shape[1]
    new_features_train=[]
    featuresIndex=[*range(D)] 
    IndexAdd=[] 
    k=0
    maxValue=0
    while k<D: 
        accs=[]
        index_d=[]
        for d in featuresIndex:  
            new_features_train.append(features_train[:,d])
            new_features_train=np.array(new_features_train)
            if k==0:
                new_features_train = new_features_train.reshape((-1, 1))
            else:
                new_features_train=np.transpose(new_features_train)
            acc=LDA.LOOCV(new_features_train,targets_train,pLDA)
            accs.append(acc)
            index_d.append(d)
            new_features_train=np.transpose(new_features_train) 
            new_features_train=list(new_features_train)
            new_features_train.pop(k)         
        max_prev=maxValue
        maxValue=max(accs)
        if maxValue<max_prev:
            break
        index=accs.index(maxValue)
        index=index_d[index]
        IndexAdd.append(index) 
        new_features_train.append(features_train[:,index])
        featuresIndex.remove(index)
        k+=1
        while k>2:
            D_new=len(new_features_train)
            new_featuresIndex=[*range(D_new)] 
            accs=[]
            index_d=[]
            for d in new_featuresIndex: 
                d_new=IndexAdd[d]
                new_features_train.pop(0)
                new_features_train=np.array(new_features_train)
                new_features_train=np.transpose(new_features_train)
                acc=LDA.LOOCV(new_features_train,targets_train,pLDA)
                accs.append(acc)
                index_d.append(d_new)
                new_features_train=np.transpose(new_features_train) 
                new_features_train=list(new_features_train)
                new_features_train.append(features_train[:,d_new])
            maxValueRemove=max(accs)
            if maxValueRemove<=maxValue:
                break
            maxValue=maxValueRemove
            index_remv=accs.index(maxValueRemove)
            index=index_d[index_remv]
            featuresIndex.append(index) 
            new_features_train.pop(index_remv)
            IndexAdd.pop(index_remv)
            k-=1
    new_features_train=np.array(new_features_train)
    new_features_train=np.transpose(new_features_train)

    return new_features_train


def SFS(features_train, features_test, targets_train, targets_test, K, SFFS, **algorithm):
    if SFFS:
        new_features_train, new_features_test=SFFS_wrapper(features_train, features_test, targets_train, targets_test, K, **algorithm)
    else:
        new_features_train, new_features_test=SFS_wrapper(features_train, features_test, targets_train, targets_test, K, **algorithm)
    return new_features_train, new_features_test


def normalization(features_entry):
    """
    Realiza la normalización de las características de entrada

    Parameters
    ----------
    features_entry : array
        Vectores de características

    Returns
    -------
    features_entry : array
        Vectores de características normalizados
    """
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
    
