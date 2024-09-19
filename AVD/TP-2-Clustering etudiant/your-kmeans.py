#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# M1 Science et Ingénieurie des données
# Université de Rouen Normandie
# T. Paquet
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from numpy.linalg import norm

colors =['r','b','g','c','m','o']
n_colors = 6

###########################################################################
# initialisation ++ du kmeans
#
def initPlusPlus(X,K):
    N,p = np.shape(X)
    C = np.zeros((p,K))
    generator = np.random.default_rng()
    
    index = np.random.choice(N, 1,replace = False)
    liste_index = [index]
    C[:,0] = X[index,:]
    X = np.delete(X,index,0)
    print("k=0 C[k]=",C[:,0],"index=",index)
    k=1
    
    while k < K:
        
        # calcul des distances
        NN = X.shape[0]
        dist = np.zeros(NN)
        
       # ICI ..... 

        # calcul des probabilités

       # ICI ....
        
        # tirage aléatoire selon proba

       # ICI ...........
    return C




def my_kmeans(X,K,Visualisation=False,Seuil=0.001,Max_iterations = 100):
    
    N,p = np.shape(X)
    iteration = 0        
    Dist=np.zeros((K,N))
    J=np.zeros(Max_iterations+1)
    J[0] = 10000000
    
    # Initialisation des clusters
    # par tirage de K exemples, pour tomber dans les données     
 
    Index_init = np.random.choice(N, K,replace = False)
    C = np.zeros((p,K))
    for k in range(K):
        C[:,k] = X[Index_init[k],:].T 
        
        
    while iteration < Max_iterations:
        iteration +=1
        #################################################################
        # E step : estimation des données manquantes 
        #          affectation des données aux clusters les plus proches
        for k in range(K):
            Dist[k,:]=np.square(np.linalg.norm(X-C[:,k],axis=1))
        y=np.argmin(Dist,axis=0)
        #################################################################
        # M Step : calcul des meilleurs centres          
  
        #################################################################
        # test du critère d'arrêt l'évolution du critère est inférieure 
        # au Seuil en pour ceent
        J[iteration]=np.sum(np.min(Dist[y,:],axis=0))/N

        for k in range(K):
              C[:,k]=np.mean(X[y==k,:])
    return C, y, J[1:iteration]


if __name__ == '__main__':

#########################################################
#''' K means '''
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    K= 4


    fig = plt.figure(2, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[0:50, 0], X[0:50, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[0])
    plt.scatter(X[50:100, 0], X[50:100, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[1])
    plt.scatter(X[100:150, 0], X[100:150, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[2])
    
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.legend(scatterpoints=1)


    Cluster, y, Critere = my_kmeans(iris.data,K,Visualisation = False)
    
    Cluster, y, Critere = my_kmeans(iris.data,K,Visualisation = False)
    
    fig = plt.figure(3, figsize=(8, 6))
    for k in range(K):
        plt.plot(X[y==k, 0], X[y==k, 1], colors[k%n_colors]+'o')
    plt.plot(Cluster[0, :], Cluster[1, :],'kx')
    plt.title('K moyennes ('+str(K)+')')
    plt.show()
    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(Critere, 'o-')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Evolution du critère')
    plt.show()
        
