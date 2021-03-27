# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:45:53 2020

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
    

def targets_conv(targets,activation_function="sigmoid"):
    """
    Realiza la conversión de los vectores de clases en forma escalar a discreta

    Parameters
    ----------
    targets : array
        Vector con las clases del conjunto
    activation_function : string, optional
        Función de activación que se utilizará. The default is "sigmoid".

    Returns
    -------
    list_targets : List
        Lista de vectores con las clases en forma discreta
    """
    C,OmegaC=indexClass(targets)
    N=targets.shape[0]
    C=C+1
    list_targets=[]
    for n in range(N):
        a=np.zeros(C)
        if activation_function=="tanh":
            a-=1
        for c in range(C):
            if targets[n]==c:
                a[c]=1
        list_targets.append(a)
    return list_targets

def sigmoid(a):
    z=1/(1+np.exp(-a))
    return z

def tanh(a):
    z=np.tanh(a)
    return z  

def sigmoid_dev(a):
    z=(1/(1+np.exp(-a)))*(1-1/(1+np.exp(-a)))
    return z

def tanh_dev(a):
    z=1-np.tanh(a)*np.tanh(a)
    return z  

def a_cal(feature_vector, theta):
    a=np.matmul(np.transpose(theta),feature_vector)
    return a

def hidden_layer(units,entry_layer,theta,func):
    z=[1.0]
    a_layer=[]
    for unit in range (units):
        a=a_cal(entry_layer,theta[unit,:])
        a_layer.append(a)
        z.append(func(a))
    z=np.array(z)
    a_layer=np.array(a_layer)
    return z, a_layer


def classification_train(features_entry, theta,layers,units,func):
    """
    Realiza la clasificación especifica para tareas de entrenamiento (regresa parametros extra a una clasificación normal)

    Parameters
    ----------
    features_entry : array
        Matriz con vectores de características
    theta : list
        Lista con los vectores de peso en cada una de las capas
    layers : int
        Número de capas
    units : list
        Lista con el número de unidades en cada capa
    func : function
        función de activación a utilizar

    Returns
    -------
    prediction : list
        Lista con los vectores de las clases predichas
    list_entry : list
        Lista con las entradas de cada capa
    a_list : list
        Lista con las activaciones de cada capa
    """
    list_entry=[]
    a_list=[]
    entry_layer=features_entry
    list_entry.append(entry_layer)
    for layer in range (layers):
        entry_layer,a_layer=hidden_layer(units[layer],entry_layer,theta[layer],func)
        list_entry.append(entry_layer)
        a_list.append(a_layer)
    entry_layer=list(entry_layer)
    entry_layer.pop(0)
    entry_layer=np.array(entry_layer)
    prediction=entry_layer
    list_entry.pop()
    return prediction,list_entry,a_list

def train_stochastic(features_train,targets_train,hidden_layers,units, iterations, rate, activation_function, **reg):
    """
    Se realiza el entrenamiento por medio de un descenso de gradiente estocástico

    Parameters
    ----------
    features_train : array
        Matriz con vectores de características de entrenamiento
    targets_train : array
        Vectores con las clases de cada muestra
    hidden_layers : int
        Número de capas ocultas
    units : list
        Lista con las unidades en cada capa oculta
    iterations : int
        Número de iteraciones a realizar
    rate : float
        Tasa de aprendizaje utilizada
    activation_function : string
        Función de activación
    **reg : tuple
        Parametros de regularización (bool,float)

    Returns
    -------
    theta: list
        Lista con los vectores de los pesos
    """
    if activation_function=="sigmoid":
        func=sigmoid
        func_dev=sigmoid_dev
    elif activation_function=="tanh":
        func=tanh
        func_dev=tanh_dev
    else:
        print("Función de activación invalida")
        return 0    
    C,OmegaC=indexClass(targets_train)
    targets_train=targets_conv(targets_train,activation_function)
    b = np.ones((features_train.shape[0],features_train.shape[1]+1))
    b[:,:-1] = features_train
    features_train=b
    units.append(C+1)
    layers=hidden_layers+1
    d=features_train.shape[1]
    N=features_train.shape[0]
    theta=[]
    for layer in range(layers):
        if layer==0:
            theta.append(np.random.rand(units[layer],d))
        else:
            theta.append(np.random.rand(units[layer],units[layer-1]+1))
    for i in range(iterations):
        for n in range(N):
            list_grad_theta=[]
            prediction,list_entry,a_list=classification_train(features_train[n,:],theta,layers,units,func)
            a=a_list[layers-1]
            if activation_function=="sigmoid":
                delta_k=(prediction-targets_train[n])*(1/(1+np.exp(-a)))*(1-1/(1+np.exp(-a)))
            elif activation_function=="tanh":
                delta_k=(prediction-targets_train[n])*(1-np.tanh(a)*np.tanh(a))   
            else:
                return 0
            #delta_k=(prediction-targets_train[n])*func_dev(a)
            delta_k= delta_k.reshape((-1, 1))
            current_entry=list_entry[layers-1]
            current_entry=current_entry.reshape((1,-1))
            Delta_Ek=np.matmul(delta_k,current_entry)
            list_grad_theta.append(Delta_Ek)
            delta=delta_k
            for layer in range(layers-2,-1,-1):
                a=a_list[layer]
                theta_new=list(np.transpose(theta[layer+1]))
                theta_new.pop()
                theta_new=np.array(theta_new)
                m=np.matmul(theta_new,delta)
                m=m.flatten()
                a=a.flatten()
                delta_j= m*func_dev(a)
                delta_j=delta_j.reshape((-1,1))
                current_entry=list_entry[layer]
                current_entry=current_entry.reshape((1,-1))
                Delta_Ej=np.matmul(delta_j,current_entry)
                list_grad_theta.append(Delta_Ej)
                delta=delta_j
            list_grad_theta.reverse()
            for layer in range(layers):
                if reg["Regularization"]:
                    w=theta[layer]
                    w[:,w.shape[1]-1]=np.zeros(w.shape[0])
                    theta[layer]=theta[layer]-rate*(list_grad_theta[layer]+ reg["lamb"]*w)
                else:
                    theta[layer]=theta[layer]-rate*list_grad_theta[layer]
    return theta

def train_batch (features_train,targets_train,hidden_layers,units, iterations, rate, activation_function, **reg):
    """
    Se realiza el entrenamiento por medio de un descenso de gradiente por lotes

    Parameters
    ----------
    features_train : array
        Matriz con vectores de características de entrenamiento
    targets_train : array
        Vectores con las clases de cada muestra
    hidden_layers : int
        Número de capas ocultas
    units : list
        Lista con las unidades en cada capa oculta
    iterations : int
        Número de iteraciones a realizar
    rate : float
        Tasa de aprendizaje utilizada
    activation_function : string
        Función de activación
    **reg : tuple
        Parametros de regularización (bool,float)

    Returns
    -------
    theta: list
        Lista con los vectores de los pesos
    """
    if activation_function=="sigmoid":
        func=sigmoid
        func_dev=sigmoid_dev
    elif activation_function=="tanh":
        func=tanh
        func_dev=tanh_dev
    else:
        print("Función de activación invalida")
        return 0    
    C,OmegaC=indexClass(targets_train)
    targets_train=targets_conv(targets_train,activation_function)
    b = np.ones((features_train.shape[0],features_train.shape[1]+1))
    b[:,:-1] = features_train
    features_train=b
    units.append(C+1)
    layers=hidden_layers+1
    d=features_train.shape[1]
    N=features_train.shape[0]
    theta=[]
    for layer in range(layers):
        if layer==0:
            theta.append(np.random.rand(units[layer],d))
        else:
            theta.append(np.random.rand(units[layer],units[layer-1]+1))
    for i in range(iterations):
        list_grad_t=[]
        for n in range(N):
            list_grad_theta=[]
            prediction,list_entry,a_list=classification_train(features_train[n,:],theta,layers,units,func)
            a=a_list[layers-1]
            delta_k=(prediction-targets_train[n])*func_dev(a)
            delta_k= delta_k.reshape((-1, 1))
            current_entry=list_entry[layers-1]
            current_entry=current_entry.reshape((1,-1))
            Delta_Ek=np.matmul(delta_k,current_entry)
            list_grad_theta.append(Delta_Ek)
            delta=delta_k
            for layer in range(layers-2,-1,-1):
                a=a_list[layer]
                theta_new=list(np.transpose(theta[layer+1]))
                theta_new.pop()
                theta_new=np.array(theta_new)
                m=np.matmul(theta_new,delta)
                m=m.flatten()
                a=a.flatten()
                delta_j=func_dev(a)*m
                delta_j=delta_j.reshape((-1,1))
                current_entry=list_entry[layer]
                current_entry=current_entry.reshape((1,-1))
                Delta_Ej=np.matmul(delta_j,current_entry)
                list_grad_theta.append(Delta_Ej)
                delta=delta_j
            list_grad_t.append(list_grad_theta)
        list_grad_prom=[]
        for layer in range(layers):
            grad_temp=np.zeros(list_grad_t[0][layer].shape)
            for n in range(N):
                grad_temp+=list_grad_t[n][layer]
            grad_temp/=N
            list_grad_prom.append(grad_temp)
        list_grad_prom.reverse()
        for layer in range(layers):
            if reg["Regularization"]:
                w=theta[layer]
                w[:,w.shape[1]-1]=np.zeros(w.shape[0])
                theta[layer]=theta[layer]-rate*(list_grad_prom[layer]+reg["lamb"]*w)
            else:
                theta[layer]=theta[layer]-rate*list_grad_prom[layer]     
    return theta


def train(features_train,targets_train,hidden_layers,units, iterations, rate, activation_function="sigmoid", Type="batch",**reg):
    """
    Función para realizar el entrenamiento del MLP

    Parameters
    ----------
    features_train : array
        Matriz con vectores de características de entrenamiento
    targets_train : array
        Vectores con las clases de cada muestra
    hidden_layers : int
        Número de capas ocultas
    units : list
        Lista con las unidades en cada capa oculta
    iterations : int
        Número de iteraciones a realizar
    rate : float
        Tasa de aprendizaje utilizada
    activation_function : string, optional
        Función de activación por utilizar. The default is "sigmoid".
    Type : string, optional
        Tipo de descenso de gradiente. The default is "batch".
    **reg : tuple
        Parametros de regularización (bool,float)

    Returns
    -------
    theta: list
        Lista con los vectores de los pesos
    """
    if Type=="batch":
        theta=train_batch(features_train,targets_train,hidden_layers,units, iterations, rate, activation_function, **reg)
    elif Type=="stochastic":
        theta=train_stochastic(features_train,targets_train,hidden_layers,units, iterations, rate, activation_function,**reg)
    else:
        print("Método de entrenamiento no válido")
        return 0
    return theta


def classification(features_entry,theta,activation_function="sigmoid"):
    """
    Función para realizar una clasificación de un conjunto de características

    Parameters
    ----------
    features_entry : array
        Matriz con los vectores de características de entrada
    theta : list
        Lista con la matriz de vectores de pesos
    activation_function : string
        Tipo de función de activación

    Returns
    -------
    prediction: list
        Lista con vectores correspondientes a las clases determinadas
    """
    if activation_function=="sigmoid":
        func=sigmoid
    elif activation_function=="tanh":
        func=tanh
    else:
        print("Función de activación invalida")
        return 0    
    layers=len(theta)
    units=[]
    for layer in range(layers):
        units.append(theta[layer].shape[0])
    N=features_entry.shape[0]
    prediction=[]
    b = np.ones((features_entry.shape[0],features_entry.shape[1]+1))
    b[:,:-1] = features_entry
    features_entry=b
    for n in range(N):    
        entry_layer=features_entry[n,:]
        for layer in range (layers):
            entry_layer,a_layer=hidden_layer(units[layer],entry_layer,theta[layer],func)
        entry_layer=list(entry_layer)
        entry_layer.pop(0)
        entry_layer=np.array(entry_layer)
        prediction_temp=entry_layer
        prediction_temp=np.round(prediction_temp)
        prediction.append(prediction_temp)
    return prediction

def accuracy(prediction, targets):
    """
    Calcula la exactitud de la clasificación con un conjunto de prueba

    Parameters
    ----------
    prediction : List
        Lista con los vectores de las clases predichas
    targets : List
        VLista con los vectores de las clases verdaderas

    Returns
    -------
    float
        Exactitud de 0 a 1
    """
    N=len(prediction)
    i=0
    for n in range(N):
        compare=prediction[n]==targets[n]
        if compare.all():
            i+=1
    i=i/N
    return i

