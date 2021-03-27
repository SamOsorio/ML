# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:28:50 2020

@author: Samuel Osorio Gutiérrez
"""
import ML.FLD as FLD
import ML.SFFS as SFFS
import ML.LDA as LDA
from time import time
import random
import numpy as np
import pandas as pd

np.random.seed(123)

def train(features_csv,targets_csv):    

    #Features and targets import
    data_features=pd.read_csv(features_csv)
    data_targets=pd.read_csv(targets_csv)
    train_len=int((len(data_features))*0.8)
    test_len=int(len(data_features)-train_len)
    
    dim=len(data_features.columns)
    targets_len=len(data_targets.columns)
    samples=len(data_features)
    
    features_train=np.zeros((train_len,dim))
    features_test=np.zeros((test_len,dim))
    
    targets_train=np.zeros((train_len,targets_len))
    targets_test=np.zeros((test_len,targets_len))
    
    keys_features=data_features.keys()
    keys_targets=data_targets.keys()
    
    #Random arrangement of data
    total_data=[]
    for j in range (dim):
        total_data.append(data_features[keys_features[j]])
        
    for j in range (targets_len):
        total_data.append(data_targets[keys_targets[j]])
            
    random_set=list(zip(*total_data))
    random.shuffle(random_set)
    total_data=list(zip(*random_set))
    
    data_features={}
    data_targets={}
    
    for j in range (dim):
        data_features[keys_features[j]]=total_data[j]
    
    i=0
    for j in range (dim,targets_len+dim):
        data_targets[keys_targets[i]]=total_data[j]
        i+=1
            
    #Create training set and test set 
    for j in range (dim):
        for i in range(train_len):        
            features_train[i][j]=data_features[keys_features[j]][i]
        k=0
        for i in range(train_len,samples):
            features_test[k][j]=data_features[keys_features[j]][i]
            k+=1
        
    for j in range (targets_len):
        for i in range(train_len):        
            targets_train[i][j]=data_targets[keys_targets[j]][i]
        k=0
        for i in range(train_len,samples):
            targets_test[k][j]=data_targets[keys_targets[j]][i]
            k+=1
            
    targets_train=targets_train.astype(int)
    targets_test=targets_test.astype(int)
    
    #Neutrality
    
    #SFS_1
    t_start=time()
    SFS_features_train_1, SFS_features_test_1,index_cod_1 =SFFS.SFFS_wrapper(features_train, features_test, targets_train[:,0], targets_test[:,0], pLDA=True)
    t_final=time()
    
    print("\x1b[1;37m"+"\nExecution time of SFFS_1 with "+ str(SFS_features_train_1.shape[1])+ " dimensions: " + str((t_final-t_start)*1000) + " ms")
    
    #Training set and test set union
    SFS_features_train_1=list(SFS_features_train_1)
    SFS_features_test_1=list(SFS_features_test_1)
    targets_train=list(targets_train)
    targets_test=list(targets_test)
    
    SFS_features_temp= SFS_features_train_1+SFS_features_test_1
    targets_temp=targets_train+targets_test
    
    SFS_features_temp=np.array(SFS_features_temp)
    targets_temp=np.array(targets_temp)
    targets_train=np.array(targets_train)
    targets_test=np.array(targets_test)
    
    #FLD_1
    K_FLD=1
    t_start=time()
    matrix_W_1=FLD.train(SFS_features_temp,targets_temp[:,0],K_FLD, pLDA=True, plot=False, Terminal=False)
    t_final=time()
    print("\nExecution time of FLD_1 with "+ str(K_FLD)+ " dimensions: " + str((t_final-t_start)*1000) + " ms")
    
    new_features_train_1=np.matmul(SFS_features_temp,matrix_W_1)
    
    #LDA_1
    print("\x1b[1;33m"+"\n*********************LDA_1*********************")
    
    t_start=time()
    S_inv_1,m_k_1,pi_k_1=LDA.train(new_features_train_1,targets_temp[:,0], pLDA=False, Terminal=False)
    t_final=time()
    print("\x1b[1;37m"+"\nTraining time: "  + str((t_final-t_start)*1000) + " ms")
    acc=LDA.LOOCV(new_features_train_1,targets_temp[:,0],pLDA=False)
    print("\nAccuracy: " + str(acc*100) + " %")

    #Discard neutral samples
    features_train=list(features_train)
    features_test=list(features_test)
    targets_train=list(targets_train)
    targets_test=list(targets_test)
    
    i=0
    while i<len(targets_train): 
        if targets_train[i][0]==1:
            targets_train.pop(i)
            features_train.pop(i)
            i-=1
        i+=1
    i=0
    while i<len(targets_test):
        if targets_test[i][0]==1:
            targets_test.pop(i)
            features_test.pop(i)
            i-=1
        i+=1
    
    features_train=np.array(features_train)
    features_test=np.array(features_test)
    targets_train=np.array(targets_train)
    targets_test=np.array(targets_test)
    
    #Arousal level
    
    #SFS_2
    t_start=time()
    SFS_features_train_2, SFS_features_test_2,index_cod_2 =SFFS.SFFS_wrapper(features_train, features_test, targets_train[:,1], targets_test[:,1], pLDA=True)
    t_final=time()
    
    print("\x1b[1;37m"+"\n\nExecution time of SFFS_2 with "+ str(SFS_features_train_2.shape[1])+ " dimensions: " + str((t_final-t_start)*1000) + " ms")
    
    #Training set and test set union
    SFS_features_train_2=list(SFS_features_train_2)
    SFS_features_test_2=list(SFS_features_test_2)
    targets_train=list(targets_train)
    targets_test=list(targets_test)
    
    SFS_features_temp= SFS_features_train_2+SFS_features_test_2
    targets_temp=targets_train+targets_test
    
    SFS_features_temp=np.array(SFS_features_temp)
    targets_temp=np.array(targets_temp)
    targets_train=np.array(targets_train)
    targets_test=np.array(targets_test)
      
    #FLD_2
    K_FLD=1
    t_start=time()
    matrix_W_2=FLD.train(SFS_features_temp,targets_temp[:,1],K_FLD, pLDA=True, plot=False, Terminal=False)
    t_final=time()
    print("\nExecution time of FLD_2 with "+ str(K_FLD)+ " dimensions: " + str((t_final-t_start)*1000) + " ms")
    
    new_features_train_2=np.matmul(SFS_features_temp,matrix_W_2)
  
    #LDA_2
    print("\x1b[1;33m"+"\n*********************LDA_2*********************")
    
    t_start=time()
    S_inv_2,m_k_2,pi_k_2=LDA.train(new_features_train_2,targets_temp[:,1], pLDA=False, Terminal=False)
    t_final=time()
    print("\x1b[1;37m"+"\nTraining time: "  + str((t_final-t_start)*1000) + " ms")
    acc=LDA.LOOCV(new_features_train_2,targets_temp[:,1],pLDA=False)
    print("\nAccuracy: " + str(acc*100) + " %")

    #Discard low arousal samples
    features_trainHA=list(features_train)
    features_testHA=list(features_test)
    targets_trainHA=list(targets_train)
    targets_testHA=list(targets_test)
    
    i=0
    while i<len(targets_trainHA): 
        if targets_trainHA[i][1]==0:
            targets_trainHA.pop(i)
            features_trainHA.pop(i)
            i-=1
        i+=1
    i=0
    while i<len(targets_testHA):
        if targets_testHA[i][1]==0:
            targets_testHA.pop(i)
            features_testHA.pop(i)
            i-=1
        i+=1
    
    features_trainHA=np.array(features_trainHA)
    features_testHA=np.array(features_testHA)
    targets_trainHA=np.array(targets_trainHA)
    targets_testHA=np.array(targets_testHA)
    
    #Valence level (high arousal)
    
    #SFS_3
    t_start=time()
    SFS_features_train_3, SFS_features_test_3,index_cod_3 =SFFS.SFFS_wrapper(features_trainHA, features_testHA, targets_trainHA[:,2], targets_testHA[:,2], pLDA=True)
    t_final=time()
    
    print("\x1b[1;37m"+"\n\nExecution time of SFFS_3 with "+ str(SFS_features_train_3.shape[1])+ " dimensions: " + str((t_final-t_start)*1000) + " ms")
    
    #Training set and test set union
    SFS_features_train_3=list(SFS_features_train_3)
    SFS_features_test_3=list(SFS_features_test_3)
    targets_trainHA=list(targets_trainHA)
    targets_testHA=list(targets_testHA)
    
    SFS_features_temp= SFS_features_train_3+SFS_features_test_3
    targets_temp=targets_trainHA+targets_testHA
    
    SFS_features_temp=np.array(SFS_features_temp)
    targets_temp=np.array(targets_temp)
       
    #FLD_3
    K_FLD=1
    t_start=time()
    matrix_W_3=FLD.train(SFS_features_temp,targets_temp[:,2],K_FLD, pLDA=True, plot=False, Terminal=False)
    t_final=time()
    print("\nExecution time of FLD_3 with "+ str(K_FLD)+ " dimensions: " + str((t_final-t_start)*1000) + " ms")
    
    new_features_train_3=np.matmul(SFS_features_temp,matrix_W_3)
    
    #LDA_3
    print("\x1b[1;33m"+"\n*********************LDA_3*********************")
    
    t_start=time()
    S_inv_3,m_k_3,pi_k_3=LDA.train(new_features_train_3,targets_temp[:,2], pLDA=False, Terminal=False)
    t_final=time()
    print("\x1b[1;37m"+"\nTraining time: "  + str((t_final-t_start)*1000) + " ms")
    acc=LDA.LOOCV(new_features_train_3,targets_temp[:,2],pLDA=False)
    print("\nAccuracy: " + str(acc*100) + " %")
    
    #Discard high arousal samples    
    features_trainLA=list(features_train)
    features_testLA=list(features_test)
    targets_trainLA=list(targets_train)
    targets_testLA=list(targets_test)
    
    i=0
    while i<len(targets_trainLA): 
        if targets_trainLA[i][1]==1:
            targets_trainLA.pop(i)
            features_trainLA.pop(i)
            i-=1
        i+=1
    i=0
    while i<len(targets_testLA):
        if targets_testLA[i][1]==1:
            targets_testLA.pop(i)
            features_testLA.pop(i)
            i-=1
        i+=1
    
    features_trainLA=np.array(features_trainLA)
    features_testLA=np.array(features_testLA)
    targets_trainLA=np.array(targets_trainLA)
    targets_testLA=np.array(targets_testLA)
    
    #Valence level (low arousal)
    
    #SFS_4
    t_start=time()
    SFS_features_train_4, SFS_features_test_4,index_cod_4 =SFFS.SFFS_wrapper(features_trainLA, features_testLA, targets_trainLA[:,2], targets_testLA[:,2], pLDA=True)
    t_final=time()
    
    print("\x1b[1;37m"+"\n\nExecution time of SFFS_4 with "+ str(SFS_features_train_4.shape[1])+ " dimensions: " + str((t_final-t_start)*1000) + " ms")
    
    #Training set and test set union
    SFS_features_train_4=list(SFS_features_train_4)
    SFS_features_test_4=list(SFS_features_test_4)
    targets_trainLA=list(targets_trainLA)
    targets_testLA=list(targets_testLA)
    
    SFS_features_temp= SFS_features_train_4+SFS_features_test_4
    targets_temp=targets_trainLA+targets_testLA
    
    SFS_features_temp=np.array(SFS_features_temp)
    targets_temp=np.array(targets_temp)
       
    #FLD_4
    K_FLD=1
    t_start=time()
    matrix_W_4=FLD.train(SFS_features_temp,targets_temp[:,2],K_FLD, pLDA=True, plot=False, Terminal=False)
    t_final=time()
    print("\nExecution time of FLD_4 with "+ str(K_FLD)+ " dimensions: " + str((t_final-t_start)*1000) + " ms")
    
    new_features_train_4=np.matmul(SFS_features_temp,matrix_W_4)
   
    #LDA_4
    print("\x1b[1;33m"+"\n*********************LDA_4*********************")
    
    t_start=time()
    S_inv_4,m_k_4,pi_k_4=LDA.train(new_features_train_4,targets_temp[:,2], pLDA=False, Terminal=False)
    t_final=time()
    print("\x1b[1;37m"+"\nTraining time: "  + str((t_final-t_start)*1000) + " ms")
    acc=LDA.LOOCV(new_features_train_4,targets_temp[:,2],pLDA=False)
    print("\nAccuracy: " + str(acc*100) + " %\n")


    del dim,i,j,k, keys_features, keys_targets, random_set, samples, t_final, t_start, targets_len, test_len, total_data, train_len,SFS_features_temp, targets_temp
       
    #pik to ln(pik)
    pi_k_1=np.log(pi_k_1)
    pi_k_2=np.log(pi_k_2)
    pi_k_3=np.log(pi_k_3)
    pi_k_4=np.log(pi_k_4)
    
    print(f"W1={matrix_W_1} \n W2={matrix_W_2} \n W3={matrix_W_3} \n W4={matrix_W_4}")
    print(f"mk1={m_k_1} \n mk2={m_k_2} \n mk3={m_k_3} \n mk4={m_k_4}")
    print(f"pik1={pi_k_1} \n pik2={pi_k_2} \n pik3={pi_k_3} \n pik4={pi_k_4}")
    print(f"S1={S_inv_1} \n S2={S_inv_2} \n S3={S_inv_3} \n S4={S_inv_4}")
    
    #Collection of parameters
    #SFFS
    list_SFFS=[index_cod_1,index_cod_2,index_cod_3,index_cod_4]
    
    Q=16
    #FLD
    matrix_W_1=matrix_W_1*2**Q
    matrix_W_1=np.int32(matrix_W_1)
    list_W_1=[]
    matrix_W_temp=matrix_W_1&0xff
    list_W_1.append(matrix_W_temp)
    matrix_W_temp=matrix_W_1&0xff00
    matrix_W_temp=np.right_shift(matrix_W_temp,8)
    list_W_1.append(matrix_W_temp)
    matrix_W_temp=matrix_W_1&0xff0000
    matrix_W_temp=np.right_shift(matrix_W_temp,16)
    list_W_1.append(matrix_W_temp)
    matrix_W_temp=matrix_W_1&0xff000000
    matrix_W_temp=np.right_shift(matrix_W_temp,24)
    list_W_1.append(matrix_W_temp)
    
    matrix_W_2=matrix_W_2*2**Q
    matrix_W_2=np.int32(matrix_W_2)
    list_W_2=[]
    matrix_W_temp=matrix_W_2&0xff
    list_W_2.append(matrix_W_temp)
    matrix_W_temp=matrix_W_2&0xff00
    matrix_W_temp=np.right_shift(matrix_W_temp,8)
    list_W_2.append(matrix_W_temp)
    matrix_W_temp=matrix_W_2&0xff0000
    matrix_W_temp=np.right_shift(matrix_W_temp,16)
    list_W_2.append(matrix_W_temp)
    matrix_W_temp=matrix_W_2&0xff000000
    matrix_W_temp=np.right_shift(matrix_W_temp,24)
    list_W_2.append(matrix_W_temp)
    
    matrix_W_3=matrix_W_3*2**Q
    matrix_W_3=np.int32(matrix_W_3)
    list_W_3=[]
    matrix_W_temp=matrix_W_3&0xff
    list_W_3.append(matrix_W_temp)
    matrix_W_temp=matrix_W_3&0xff00
    matrix_W_temp=np.right_shift(matrix_W_temp,8)
    list_W_3.append(matrix_W_temp)
    matrix_W_temp=matrix_W_3&0xff0000
    matrix_W_temp=np.right_shift(matrix_W_temp,16)
    list_W_3.append(matrix_W_temp)
    matrix_W_temp=matrix_W_3&0xff000000
    matrix_W_temp=np.right_shift(matrix_W_temp,24)
    list_W_3.append(matrix_W_temp)
    
    matrix_W_4=matrix_W_4*2**Q
    matrix_W_4=np.int32(matrix_W_4)
    list_W_4=[]
    matrix_W_temp=matrix_W_4&0xff
    list_W_4.append(matrix_W_temp)
    matrix_W_temp=matrix_W_4&0xff00
    matrix_W_temp=np.right_shift(matrix_W_temp,8)
    list_W_4.append(matrix_W_temp)
    matrix_W_temp=matrix_W_4&0xff0000
    matrix_W_temp=np.right_shift(matrix_W_temp,16)
    list_W_4.append(matrix_W_temp)
    matrix_W_temp=matrix_W_4&0xff000000
    matrix_W_temp=np.right_shift(matrix_W_temp,24)
    list_W_4.append(matrix_W_temp)
    
    list_FLD=[list_W_1,list_W_2,list_W_3,list_W_4]
    
    #LDA
    #mk
    m_k_1=m_k_1*2**Q
    m_k_1=np.int32(m_k_1)
    list_m_k_1=[]
    m_k_temp=m_k_1&0xff
    list_m_k_1.append(m_k_temp)
    m_k_temp=m_k_1&0xff00
    m_k_temp=np.right_shift(m_k_temp,8)
    list_m_k_1.append(m_k_temp)
    m_k_temp=m_k_1&0xff0000
    m_k_temp=np.right_shift(m_k_temp,16)
    list_m_k_1.append(m_k_temp)
    m_k_temp=m_k_1&0xff000000
    m_k_temp=np.right_shift(m_k_temp,24)
    list_m_k_1.append(m_k_temp)
    
    m_k_2=m_k_2*2**Q
    m_k_2=np.int32(m_k_2)
    list_m_k_2=[]
    m_k_temp=m_k_2&0xff
    list_m_k_2.append(m_k_temp)
    m_k_temp=m_k_2&0xff00
    m_k_temp=np.right_shift(m_k_temp,8)
    list_m_k_2.append(m_k_temp)
    m_k_temp=m_k_2&0xff0000
    m_k_temp=np.right_shift(m_k_temp,16)
    list_m_k_2.append(m_k_temp)
    m_k_temp=m_k_2&0xff000000
    m_k_temp=np.right_shift(m_k_temp,24)
    list_m_k_2.append(m_k_temp)
    
    m_k_3=m_k_3*2**Q
    m_k_3=np.int32(m_k_3)
    list_m_k_3=[]
    m_k_temp=m_k_2&0xff
    list_m_k_3.append(m_k_temp)
    m_k_temp=m_k_3&0xff00
    m_k_temp=np.right_shift(m_k_temp,8)
    list_m_k_3.append(m_k_temp)
    m_k_temp=m_k_3&0xff0000
    m_k_temp=np.right_shift(m_k_temp,16)
    list_m_k_3.append(m_k_temp)
    m_k_temp=m_k_3&0xff000000
    m_k_temp=np.right_shift(m_k_temp,24)
    list_m_k_3.append(m_k_temp)
    
    m_k_4=m_k_4*2**Q
    m_k_4=np.int32(m_k_4)
    list_m_k_4=[]
    m_k_temp=m_k_4&0xff
    list_m_k_4.append(m_k_temp)
    m_k_temp=m_k_4&0xff00
    m_k_temp=np.right_shift(m_k_temp,8)
    list_m_k_4.append(m_k_temp)
    m_k_temp=m_k_4&0xff0000
    m_k_temp=np.right_shift(m_k_temp,16)
    list_m_k_4.append(m_k_temp)
    m_k_temp=m_k_4&0xff000000
    m_k_temp=np.right_shift(m_k_temp,24)
    list_m_k_4.append(m_k_temp)
    
    list_LDA_mk=[list_m_k_1,list_m_k_2,list_m_k_3,list_m_k_4]
    
    #ln(pik)
    
    pi_k_1=pi_k_1*2**Q
    pi_k_1=np.int32(pi_k_1)
    list_pi_k_1=[]
    pi_k_temp=pi_k_1&0xff
    list_pi_k_1.append(pi_k_temp)
    pi_k_temp=pi_k_1&0xff00
    pi_k_temp=np.right_shift(pi_k_temp,8)
    list_pi_k_1.append(pi_k_temp)
    pi_k_temp=pi_k_1&0xff0000
    pi_k_temp=np.right_shift(pi_k_temp,16)
    list_pi_k_1.append(pi_k_temp)
    pi_k_temp=pi_k_1&0xff000000
    pi_k_temp=np.right_shift(pi_k_temp,24)
    list_pi_k_1.append(pi_k_temp)
    
    pi_k_2=pi_k_2*2**Q
    pi_k_2=np.int32(pi_k_2)
    list_pi_k_2=[]
    pi_k_temp=pi_k_2&0xff
    list_pi_k_2.append(pi_k_temp)
    pi_k_temp=pi_k_2&0xff00
    pi_k_temp=np.right_shift(pi_k_temp,8)
    list_pi_k_2.append(pi_k_temp)
    pi_k_temp=pi_k_2&0xff0000
    pi_k_temp=np.right_shift(pi_k_temp,16)
    list_pi_k_2.append(pi_k_temp)
    pi_k_temp=pi_k_2&0xff000000
    pi_k_temp=np.right_shift(pi_k_temp,24)
    list_pi_k_2.append(pi_k_temp)
    
    pi_k_3=pi_k_3*2**Q
    pi_k_3=np.int32(pi_k_3)
    list_pi_k_3=[]
    pi_k_temp=pi_k_3&0xff
    list_pi_k_3.append(pi_k_temp)
    pi_k_temp=pi_k_3&0xff00
    pi_k_temp=np.right_shift(pi_k_temp,8)
    list_pi_k_3.append(pi_k_temp)
    pi_k_temp=pi_k_3&0xff0000
    pi_k_temp=np.right_shift(pi_k_temp,16)
    list_pi_k_3.append(pi_k_temp)
    pi_k_temp=pi_k_3&0xff000000
    pi_k_temp=np.right_shift(pi_k_temp,24)
    list_pi_k_3.append(pi_k_temp)
    
    pi_k_4=pi_k_4*2**Q
    pi_k_4=np.int32(pi_k_4)
    list_pi_k_4=[]
    pi_k_temp=pi_k_4&0xff
    list_pi_k_4.append(pi_k_temp)
    pi_k_temp=pi_k_4&0xff00
    pi_k_temp=np.right_shift(pi_k_temp,8)
    list_pi_k_4.append(pi_k_temp)
    pi_k_temp=pi_k_4&0xff0000
    pi_k_temp=np.right_shift(pi_k_temp,16)
    list_pi_k_4.append(pi_k_temp)
    pi_k_temp=pi_k_4&0xff000000
    pi_k_temp=np.right_shift(pi_k_temp,24)
    list_pi_k_4.append(pi_k_temp)
    
    list_LDA_pik=[list_pi_k_1,list_pi_k_2,list_pi_k_3,list_pi_k_4]
    
    #S_inv
    S_inv_1=S_inv_1*2**Q
    S_inv_1=np.int32(S_inv_1)
    list_S_inv_1=[]
    S_inv_temp=S_inv_1&0xff
    list_S_inv_1.append(S_inv_temp)
    S_inv_temp=S_inv_1&0xff00
    S_inv_temp=np.right_shift(S_inv_temp,8)
    list_S_inv_1.append(S_inv_temp)
    S_inv_temp=S_inv_1&0xff0000
    S_inv_temp=np.right_shift(S_inv_temp,16)
    list_S_inv_1.append(S_inv_temp)
    S_inv_temp=S_inv_1&0xff000000
    S_inv_temp=np.right_shift(S_inv_temp,24)
    list_S_inv_1.append(S_inv_temp)
    
    S_inv_2=S_inv_2*2**Q
    S_inv_2=np.int32(S_inv_2)
    list_S_inv_2=[]
    S_inv_temp=S_inv_2&0xff
    list_S_inv_2.append(S_inv_temp)
    S_inv_temp=S_inv_2&0xff00
    S_inv_temp=np.right_shift(S_inv_temp,8)
    list_S_inv_2.append(S_inv_temp)
    S_inv_temp=S_inv_2&0xff0000
    S_inv_temp=np.right_shift(S_inv_temp,16)
    list_S_inv_2.append(S_inv_temp)
    S_inv_temp=S_inv_2&0xff000000
    S_inv_temp=np.right_shift(S_inv_temp,24)
    list_S_inv_2.append(S_inv_temp)

    S_inv_3=S_inv_3*2**Q
    S_inv_3=np.int32(S_inv_3)
    list_S_inv_3=[]
    S_inv_temp=S_inv_3&0xff
    list_S_inv_3.append(S_inv_temp)
    S_inv_temp=S_inv_3&0xff00
    S_inv_temp=np.right_shift(S_inv_temp,8)
    list_S_inv_3.append(S_inv_temp)
    S_inv_temp=S_inv_3&0xff0000
    S_inv_temp=np.right_shift(S_inv_temp,16)
    list_S_inv_3.append(S_inv_temp)
    S_inv_temp=S_inv_3&0xff000000
    S_inv_temp=np.right_shift(S_inv_temp,24)
    list_S_inv_3.append(S_inv_temp)

    S_inv_4=S_inv_4*2**Q
    S_inv_4=np.int32(S_inv_4)
    list_S_inv_4=[]
    S_inv_temp=S_inv_4&0xff
    list_S_inv_4.append(S_inv_temp)
    S_inv_temp=S_inv_4&0xff00
    S_inv_temp=np.right_shift(S_inv_temp,8)
    list_S_inv_4.append(S_inv_temp)
    S_inv_temp=S_inv_4&0xff0000
    S_inv_temp=np.right_shift(S_inv_temp,16)
    list_S_inv_4.append(S_inv_temp)
    S_inv_temp=S_inv_4&0xff000000
    S_inv_temp=np.right_shift(S_inv_temp,24)
    list_S_inv_4.append(S_inv_temp)
    
    list_LDA_S_inv=[list_S_inv_1,list_S_inv_2,list_S_inv_3,list_S_inv_4]
    
    return list_SFFS, list_FLD, list_LDA_mk,list_LDA_pik,list_LDA_S_inv
    
def train_LOOCV(features_csv,targets_csv):    

    #Features and targets import
    data_features=pd.read_csv(features_csv)
    data_targets=pd.read_csv(targets_csv)
    
    dim=len(data_features.columns)
    targets_len=len(data_targets.columns)
    samples=len(data_features)
    
    features_train=np.zeros((samples,dim)) 
    targets_train=np.zeros((samples,targets_len))

    keys_features=data_features.keys()
    keys_targets=data_targets.keys()
    
            
    #Create training set
    for j in range (dim):
        for i in range(samples):        
            features_train[i][j]=data_features[keys_features[j]][i]

        
    for j in range (targets_len):
        for i in range(samples):        
            targets_train[i][j]=data_targets[keys_targets[j]][i]
            
    targets_train=targets_train.astype(int)
    
    #Neutrality
    
    #SFS_1
    t_start=time()
    SFS_features_train_1,index_cod_1 =SFFS.SFFS_wrapper_LOOCV(features_train,targets_train[:,0], pLDA=True)
    t_final=time()
    
    print("\x1b[1;37m"+"\nExecution time of SFFS_1 with "+ str(SFS_features_train_1.shape[1])+ " dimensions: " + str((t_final-t_start)*1000) + " ms")
    
    #FLD_1
    K_FLD=1
    t_start=time()
    matrix_W_1=FLD.train(SFS_features_train_1,targets_train[:,0],K_FLD, pLDA=True, plot=False, Terminal=False)
    t_final=time()
    print("\nExecution time of FLD_1 with "+ str(K_FLD)+ " dimensions: " + str((t_final-t_start)*1000) + " ms")
    
    new_features_train_1=np.matmul(SFS_features_train_1,matrix_W_1)
    
    #LDA_1
    print("\x1b[1;33m"+"\n*********************LDA_1*********************")
    
    t_start=time()
    S_inv_1,m_k_1,pi_k_1=LDA.train(new_features_train_1,targets_train[:,0], pLDA=False, Terminal=False)
    t_final=time()
    print("\x1b[1;37m"+"\nTraining time: "  + str((t_final-t_start)*1000) + " ms")
    acc=LDA.LOOCV(new_features_train_1,targets_train[:,0],pLDA=False)
    print("\nAccuracy: " + str(acc*100) + " %")
   
    #Discard neutral samples
    features_train=list(features_train)
    targets_train=list(targets_train)
    
    i=0
    while i<len(targets_train): 
        if targets_train[i][0]==1:
            targets_train.pop(i)
            features_train.pop(i)
            i-=1
        i+=1
    
    features_train=np.array(features_train)
    targets_train=np.array(targets_train)
    
    #Arousal level
    
    #SFS_2
    t_start=time()
    SFS_features_train_2, index_cod_2 =SFFS.SFFS_wrapper_LOOCV(features_train, targets_train[:,1], pLDA=True)
    t_final=time()
    
    print("\x1b[1;37m"+"\n\nExecution time of SFFS_2 with "+ str(SFS_features_train_2.shape[1])+ " dimensions: " + str((t_final-t_start)*1000) + " ms")
    
    #FLD_2
    K_FLD=1
    t_start=time()
    matrix_W_2=FLD.train(SFS_features_train_2,targets_train[:,1],K_FLD, pLDA=True, plot=False, Terminal=False)
    t_final=time()
    print("\nExecution time of FLD_2 with "+ str(K_FLD)+ " dimensions: " + str((t_final-t_start)*1000) + " ms")
    
    new_features_train_2=np.matmul(SFS_features_train_2,matrix_W_2)
    
    #LDA_2
    print("\x1b[1;33m"+"\n*********************LDA_2*********************")
    
    t_start=time()
    S_inv_2,m_k_2,pi_k_2=LDA.train(new_features_train_2,targets_train[:,1], pLDA=False, Terminal=False)
    t_final=time()
    print("\x1b[1;37m"+"\nTraining time: "  + str((t_final-t_start)*1000) + " ms")
    acc=LDA.LOOCV(new_features_train_2,targets_train[:,1],pLDA=False)
    print("\nAccuracy: " + str(acc*100) + " %")

    #Discard low arousal samples
    
    features_trainHA=list(features_train)
    targets_trainHA=list(targets_train)
    
    i=0
    while i<len(targets_trainHA): 
        if targets_trainHA[i][1]==0:
            targets_trainHA.pop(i)
            features_trainHA.pop(i)
            i-=1
        i+=1

    features_trainHA=np.array(features_trainHA)
    targets_trainHA=np.array(targets_trainHA)
    
    #Valence level (high arousal)
    
    #SFS_3
    t_start=time()
    SFS_features_train_3,index_cod_3 =SFFS.SFFS_wrapper_LOOCV(features_trainHA, targets_trainHA[:,2], pLDA=True)
    t_final=time()
    
    print("\x1b[1;37m"+"\n\nExecution time of SFFS_3 with "+ str(SFS_features_train_3.shape[1])+ " dimensions: " + str((t_final-t_start)*1000) + " ms")
    
    #FLD_3
    K_FLD=1
    t_start=time()
    matrix_W_3=FLD.train(SFS_features_train_3,targets_trainHA[:,2],K_FLD, pLDA=True, plot=False, Terminal=False)
    t_final=time()
    print("\nExecution time of FLD_3 with "+ str(K_FLD)+ " dimensions: " + str((t_final-t_start)*1000) + " ms")
    
    new_features_train_3=np.matmul(SFS_features_train_3,matrix_W_3)
    
    #LDA_3
    print("\x1b[1;33m"+"\n*********************LDA_3*********************")
    
    t_start=time()
    S_inv_3,m_k_3,pi_k_3=LDA.train(new_features_train_3,targets_trainHA[:,2], pLDA=False, Terminal=False)
    t_final=time()
    print("\x1b[1;37m"+"\nTraining time: "  + str((t_final-t_start)*1000) + " ms")
    acc=LDA.LOOCV(new_features_train_3,targets_trainHA[:,2],pLDA=False)
    print("\nAccuracy: " + str(acc*100) + " %")

    #Discard high arousal samples
    
    features_trainLA=list(features_train)
    targets_trainLA=list(targets_train)
    
    i=0
    while i<len(targets_trainLA): 
        if targets_trainLA[i][1]==1:
            targets_trainLA.pop(i)
            features_trainLA.pop(i)
            i-=1
        i+=1
    i=0
    
    features_trainLA=np.array(features_trainLA)
    targets_trainLA=np.array(targets_trainLA)
    
    ##Valence level (low arousal)    
    #SFS_4
    t_start=time()
    SFS_features_train_4,index_cod_4 =SFFS.SFFS_wrapper_LOOCV(features_trainLA, targets_trainLA[:,2], pLDA=True)
    t_final=time()
    
    print("\x1b[1;37m"+"\n\nExecution time of SFFS_4 with "+ str(SFS_features_train_4.shape[1])+ " dimensions: " + str((t_final-t_start)*1000) + " ms")
    
    #FLD_4
    K_FLD=1
    t_start=time()
    matrix_W_4=FLD.train(SFS_features_train_4,targets_trainLA[:,2],K_FLD, pLDA=True, plot=False, Terminal=False)
    t_final=time()
    print("\nExecution time of FLD_4 with "+ str(K_FLD)+ " dimensions: " + str((t_final-t_start)*1000) + " ms")
    
    new_features_train_4=np.matmul(SFS_features_train_4,matrix_W_4)
    
    #LDA_4
    print("\x1b[1;33m"+"\n*********************LDA_4*********************")
    
    t_start=time()
    S_inv_4,m_k_4,pi_k_4=LDA.train(new_features_train_4,targets_trainLA[:,2], pLDA=False, Terminal=False)
    t_final=time()
    print("\x1b[1;37m"+"\nTraining time: "  + str((t_final-t_start)*1000) + " ms")
    acc=LDA.LOOCV(new_features_train_4,targets_trainLA[:,2],pLDA=False)
    print("\nAccuracy: " + str(acc*100) + " %\n")

    del acc,dim,i,j,K_FLD, keys_features, keys_targets, samples, t_final, t_start, targets_len
    
    #pik to ln(pik) 
    pi_k_1=np.log(pi_k_1)
    pi_k_2=np.log(pi_k_2)
    pi_k_3=np.log(pi_k_3)
    pi_k_4=np.log(pi_k_4)
    
    print(f"W1={matrix_W_1} \n W2={matrix_W_2} \n W3={matrix_W_3} \n W4={matrix_W_4}")
    print(f"mk1={m_k_1} \n mk2={m_k_2} \n mk3={m_k_3} \n mk4={m_k_4}")
    print(f"pik1={pi_k_1} \n pik2={pi_k_2} \n pik3={pi_k_3} \n pik4={pi_k_4}")
    print(f"S1={S_inv_1} \n S2={S_inv_2} \n S3={S_inv_3} \n S4={S_inv_4}")
    
    #Collection of parameters
    #SFFS
    list_SFFS=[index_cod_1,index_cod_2,index_cod_3,index_cod_4]
    
    Q=16
    #LDF
    matrix_W_1=matrix_W_1*2**Q
    matrix_W_1=np.int32(matrix_W_1)
    list_W_1=[]
    matrix_W_temp=matrix_W_1&0xff
    list_W_1.append(matrix_W_temp)
    matrix_W_temp=matrix_W_1&0xff00
    matrix_W_temp=np.right_shift(matrix_W_temp,8)
    list_W_1.append(matrix_W_temp)
    matrix_W_temp=matrix_W_1&0xff0000
    matrix_W_temp=np.right_shift(matrix_W_temp,16)
    list_W_1.append(matrix_W_temp)
    matrix_W_temp=matrix_W_1&0xff000000
    matrix_W_temp=np.right_shift(matrix_W_temp,24)
    list_W_1.append(matrix_W_temp)
    
    matrix_W_2=matrix_W_2*2**Q
    matrix_W_2=np.int32(matrix_W_2)
    list_W_2=[]
    matrix_W_temp=matrix_W_2&0xff
    list_W_2.append(matrix_W_temp)
    matrix_W_temp=matrix_W_2&0xff00
    matrix_W_temp=np.right_shift(matrix_W_temp,8)
    list_W_2.append(matrix_W_temp)
    matrix_W_temp=matrix_W_2&0xff0000
    matrix_W_temp=np.right_shift(matrix_W_temp,16)
    list_W_2.append(matrix_W_temp)
    matrix_W_temp=matrix_W_2&0xff000000
    matrix_W_temp=np.right_shift(matrix_W_temp,24)
    list_W_2.append(matrix_W_temp)
    
    matrix_W_3=matrix_W_3*2**Q
    matrix_W_3=np.int32(matrix_W_3)
    list_W_3=[]
    matrix_W_temp=matrix_W_3&0xff
    list_W_3.append(matrix_W_temp)
    matrix_W_temp=matrix_W_3&0xff00
    matrix_W_temp=np.right_shift(matrix_W_temp,8)
    list_W_3.append(matrix_W_temp)
    matrix_W_temp=matrix_W_3&0xff0000
    matrix_W_temp=np.right_shift(matrix_W_temp,16)
    list_W_3.append(matrix_W_temp)
    matrix_W_temp=matrix_W_3&0xff000000
    matrix_W_temp=np.right_shift(matrix_W_temp,24)
    list_W_3.append(matrix_W_temp)
    

    matrix_W_4=matrix_W_4*2**Q
    matrix_W_4=np.int32(matrix_W_4)
    list_W_4=[]
    matrix_W_temp=matrix_W_4&0xff
    list_W_4.append(matrix_W_temp)
    matrix_W_temp=matrix_W_4&0xff00
    matrix_W_temp=np.right_shift(matrix_W_temp,8)
    list_W_4.append(matrix_W_temp)
    matrix_W_temp=matrix_W_4&0xff0000
    matrix_W_temp=np.right_shift(matrix_W_temp,16)
    list_W_4.append(matrix_W_temp)
    matrix_W_temp=matrix_W_4&0xff000000
    matrix_W_temp=np.right_shift(matrix_W_temp,24)
    list_W_4.append(matrix_W_temp)
    
    list_FLD=[list_W_1,list_W_2,list_W_3,list_W_4]
    
    #LDA
    #mk
    m_k_1=m_k_1*2**Q
    m_k_1=np.int32(m_k_1)
    list_m_k_1=[]
    m_k_temp=m_k_1&0xff
    list_m_k_1.append(m_k_temp)
    m_k_temp=m_k_1&0xff00
    m_k_temp=np.right_shift(m_k_temp,8)
    list_m_k_1.append(m_k_temp)
    m_k_temp=m_k_1&0xff0000
    m_k_temp=np.right_shift(m_k_temp,16)
    list_m_k_1.append(m_k_temp)
    m_k_temp=m_k_1&0xff000000
    m_k_temp=np.right_shift(m_k_temp,24)
    list_m_k_1.append(m_k_temp)
    
    m_k_2=m_k_2*2**Q
    m_k_2=np.int32(m_k_2)
    list_m_k_2=[]
    m_k_temp=m_k_2&0xff
    list_m_k_2.append(m_k_temp)
    m_k_temp=m_k_2&0xff00
    m_k_temp=np.right_shift(m_k_temp,8)
    list_m_k_2.append(m_k_temp)
    m_k_temp=m_k_2&0xff0000
    m_k_temp=np.right_shift(m_k_temp,16)
    list_m_k_2.append(m_k_temp)
    m_k_temp=m_k_2&0xff000000
    m_k_temp=np.right_shift(m_k_temp,24)
    list_m_k_2.append(m_k_temp)
    
    m_k_3=m_k_3*2**Q
    m_k_3=np.int32(m_k_3)
    list_m_k_3=[]
    m_k_temp=m_k_2&0xff
    list_m_k_3.append(m_k_temp)
    m_k_temp=m_k_3&0xff00
    m_k_temp=np.right_shift(m_k_temp,8)
    list_m_k_3.append(m_k_temp)
    m_k_temp=m_k_3&0xff0000
    m_k_temp=np.right_shift(m_k_temp,16)
    list_m_k_3.append(m_k_temp)
    m_k_temp=m_k_3&0xff000000
    m_k_temp=np.right_shift(m_k_temp,24)
    list_m_k_3.append(m_k_temp)
    
    m_k_4=m_k_4*2**Q
    m_k_4=np.int32(m_k_4)
    list_m_k_4=[]
    m_k_temp=m_k_4&0xff
    list_m_k_4.append(m_k_temp)
    m_k_temp=m_k_4&0xff00
    m_k_temp=np.right_shift(m_k_temp,8)
    list_m_k_4.append(m_k_temp)
    m_k_temp=m_k_4&0xff0000
    m_k_temp=np.right_shift(m_k_temp,16)
    list_m_k_4.append(m_k_temp)
    m_k_temp=m_k_4&0xff000000
    m_k_temp=np.right_shift(m_k_temp,24)
    list_m_k_4.append(m_k_temp)
    
    list_LDA_mk=[list_m_k_1,list_m_k_2,list_m_k_3,list_m_k_4]
    
    #ln(pik)
    
    pi_k_1=pi_k_1*2**Q
    pi_k_1=np.int32(pi_k_1)
    list_pi_k_1=[]
    pi_k_temp=pi_k_1&0xff
    list_pi_k_1.append(pi_k_temp)
    pi_k_temp=pi_k_1&0xff00
    pi_k_temp=np.right_shift(pi_k_temp,8)
    list_pi_k_1.append(pi_k_temp)
    pi_k_temp=pi_k_1&0xff0000
    pi_k_temp=np.right_shift(pi_k_temp,16)
    list_pi_k_1.append(pi_k_temp)
    pi_k_temp=pi_k_1&0xff000000
    pi_k_temp=np.right_shift(pi_k_temp,24)
    list_pi_k_1.append(pi_k_temp)
    
    pi_k_2=pi_k_2*2**Q
    pi_k_2=np.int32(pi_k_2)
    list_pi_k_2=[]
    pi_k_temp=pi_k_2&0xff
    list_pi_k_2.append(pi_k_temp)
    pi_k_temp=pi_k_2&0xff00
    pi_k_temp=np.right_shift(pi_k_temp,8)
    list_pi_k_2.append(pi_k_temp)
    pi_k_temp=pi_k_2&0xff0000
    pi_k_temp=np.right_shift(pi_k_temp,16)
    list_pi_k_2.append(pi_k_temp)
    pi_k_temp=pi_k_2&0xff000000
    pi_k_temp=np.right_shift(pi_k_temp,24)
    list_pi_k_2.append(pi_k_temp)
    
    pi_k_3=pi_k_3*2**Q
    pi_k_3=np.int32(pi_k_3)
    list_pi_k_3=[]
    pi_k_temp=pi_k_3&0xff
    list_pi_k_3.append(pi_k_temp)
    pi_k_temp=pi_k_3&0xff00
    pi_k_temp=np.right_shift(pi_k_temp,8)
    list_pi_k_3.append(pi_k_temp)
    pi_k_temp=pi_k_3&0xff0000
    pi_k_temp=np.right_shift(pi_k_temp,16)
    list_pi_k_3.append(pi_k_temp)
    pi_k_temp=pi_k_3&0xff000000
    pi_k_temp=np.right_shift(pi_k_temp,24)
    list_pi_k_3.append(pi_k_temp)
    
    pi_k_4=pi_k_4*2**Q
    pi_k_4=np.int32(pi_k_4)
    list_pi_k_4=[]
    pi_k_temp=pi_k_4&0xff
    list_pi_k_4.append(pi_k_temp)
    pi_k_temp=pi_k_4&0xff00
    pi_k_temp=np.right_shift(pi_k_temp,8)
    list_pi_k_4.append(pi_k_temp)
    pi_k_temp=pi_k_4&0xff0000
    pi_k_temp=np.right_shift(pi_k_temp,16)
    list_pi_k_4.append(pi_k_temp)
    pi_k_temp=pi_k_4&0xff000000
    pi_k_temp=np.right_shift(pi_k_temp,24)
    list_pi_k_4.append(pi_k_temp)
    
    list_LDA_pik=[list_pi_k_1,list_pi_k_2,list_pi_k_3,list_pi_k_4]
    
    #S_inv
    S_inv_1=S_inv_1*2**Q
    S_inv_1=np.int32(S_inv_1)
    list_S_inv_1=[]
    S_inv_temp=S_inv_1&0xff
    list_S_inv_1.append(S_inv_temp)
    S_inv_temp=S_inv_1&0xff00
    S_inv_temp=np.right_shift(S_inv_temp,8)
    list_S_inv_1.append(S_inv_temp)
    S_inv_temp=S_inv_1&0xff0000
    S_inv_temp=np.right_shift(S_inv_temp,16)
    list_S_inv_1.append(S_inv_temp)
    S_inv_temp=S_inv_1&0xff000000
    S_inv_temp=np.right_shift(S_inv_temp,24)
    list_S_inv_1.append(S_inv_temp)
    
    S_inv_2=S_inv_2*2**Q
    S_inv_2=np.int32(S_inv_2)
    list_S_inv_2=[]
    S_inv_temp=S_inv_2&0xff
    list_S_inv_2.append(S_inv_temp)
    S_inv_temp=S_inv_2&0xff00
    S_inv_temp=np.right_shift(S_inv_temp,8)
    list_S_inv_2.append(S_inv_temp)
    S_inv_temp=S_inv_2&0xff0000
    S_inv_temp=np.right_shift(S_inv_temp,16)
    list_S_inv_2.append(S_inv_temp)
    S_inv_temp=S_inv_2&0xff000000
    S_inv_temp=np.right_shift(S_inv_temp,24)
    list_S_inv_2.append(S_inv_temp)

    S_inv_3=S_inv_3*2**Q
    S_inv_3=np.int32(S_inv_3)
    list_S_inv_3=[]
    S_inv_temp=S_inv_3&0xff
    list_S_inv_3.append(S_inv_temp)
    S_inv_temp=S_inv_3&0xff00
    S_inv_temp=np.right_shift(S_inv_temp,8)
    list_S_inv_3.append(S_inv_temp)
    S_inv_temp=S_inv_3&0xff0000
    S_inv_temp=np.right_shift(S_inv_temp,16)
    list_S_inv_3.append(S_inv_temp)
    S_inv_temp=S_inv_3&0xff000000
    S_inv_temp=np.right_shift(S_inv_temp,24)
    list_S_inv_3.append(S_inv_temp)

    S_inv_4=S_inv_4*2**Q
    S_inv_4=np.int32(S_inv_4)
    list_S_inv_4=[]
    S_inv_temp=S_inv_4&0xff
    list_S_inv_4.append(S_inv_temp)
    S_inv_temp=S_inv_4&0xff00
    S_inv_temp=np.right_shift(S_inv_temp,8)
    list_S_inv_4.append(S_inv_temp)
    S_inv_temp=S_inv_4&0xff0000
    S_inv_temp=np.right_shift(S_inv_temp,16)
    list_S_inv_4.append(S_inv_temp)
    S_inv_temp=S_inv_4&0xff000000
    S_inv_temp=np.right_shift(S_inv_temp,24)
    list_S_inv_4.append(S_inv_temp)
    
    list_LDA_S_inv=[list_S_inv_1,list_S_inv_2,list_S_inv_3,list_S_inv_4]
    
    return list_SFFS, list_FLD, list_LDA_mk,list_LDA_pik,list_LDA_S_inv

def EMDC(features_csv,targets_csv, LOOCV=False):
    if LOOCV:
        list_SFFS, list_FLD, list_LDA_mk,list_LDA_pik,list_LDA_S_inv=train_LOOCV(features_csv,targets_csv) 
    else:
        list_SFFS, list_FLD, list_LDA_mk,list_LDA_pik,list_LDA_S_inv=train(features_csv,targets_csv) 
    return list_SFFS, list_FLD, list_LDA_mk,list_LDA_pik,list_LDA_S_inv

if __name__ == "__main__":
    features_csv="features.csv"
    targets_csv="targets.csv" 
    list_SFFS, list_FLD, list_LDA_mk,list_LDA_pik,list_LDA_S_inv=EMDC(features_csv,targets_csv, LOOCV=True)    