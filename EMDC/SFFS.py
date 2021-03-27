# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:54:25 2020

@author: Samuel Osorio Gutiérrez
"""
import ML.LDA as LDA
import numpy as np

def SFFS_wrapper(features_train, features_test, targets_train, targets_test, pLDA):
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
    
    #codification
    #0-7 features
    ind_car=8
    index_cod=[]
    s=""
    for d in range(ind_car):
        if d in IndexAdd:
            s+="1"
        else:
            s+="0"
    index_cod.append(int(s,2))
    #8-11 features
    s=""
    for d in range(ind_car,ind_car+8):
        if d>=ind_car+4:
            break
        if d in IndexAdd:
            s+="1"
        else:
            s+="0"
    index_cod.append(int(s,2))
    #12-19 features
    s=""
    ind_car+=4
    for d in range(ind_car,ind_car+8):
        if d in IndexAdd:
            s+="1"
        else:
            s+="0"
    index_cod.append(int(s,2))
    ind_car+=8
    #20-21 features
    s=""
    for d in range(ind_car,ind_car+8):
        if d>=ind_car+1:
            break
        if d in IndexAdd:
            s+="1"
        else:
            s+="0"
    index_cod.append(int(s,2))

    return new_features_train, new_features_test,index_cod

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
    
    #codification
    #0-7 features
    ind_car=8
    index_cod=[]
    s=""
    for d in range(ind_car):
        if d in IndexAdd:
            s+="1"
        else:
            s+="0"
    index_cod.append(int(s,2))
    #8-11 features
    s=""
    for d in range(ind_car,ind_car+8):
        if d>=ind_car+4:
            break
        if d in IndexAdd:
            s+="1"
        else:
            s+="0"
    index_cod.append(int(s,2))
    #12-19 features
    s=""
    ind_car+=4
    for d in range(ind_car,ind_car+8):
        if d in IndexAdd:
            s+="1"
        else:
            s+="0"
    index_cod.append(int(s,2))
    ind_car+=8
    #20-21 features
    s=""
    for d in range(ind_car,ind_car+8):
        if d>=ind_car+1:
            break
        if d in IndexAdd:
            s+="1"
        else:
            s+="0"
    index_cod.append(int(s,2))

    return new_features_train, index_cod


