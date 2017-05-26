#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 19:38:02 2017

@author: quien
"""

import numpy as np;
import matplotlib.pyplot as plt;
import scipy.io as scio;

def hu_moments(I):
    
    X = np.arange(I.shape[0]).reshape((1,I.shape[0]));
    Y = np.arange(I.shape[0]).reshape((I.shape[1],1));
    
    m00 = np.sum(I);    
    X_m = np.sum(np.dot(X,I))/m00;
    Y_m = np.sum(np.dot(I,Y))/m00;
    X = X - X_m;
    Y = Y - Y_m;
    '''
    M = [];
    for i in range(K):
        for j in range(K):
            if i+j >= 2:
                M.append(np.dot(X**i,np.dot(I,Y**j))/np.power(m00,1.0+0.5*(i+j)));
    '''
    n11 = np.dot(X,np.dot(I,Y))    / np.power(m00,2.0);
    n20 = np.sum(np.dot(X**2,I))   / np.power(m00,2.0);
    n02 = np.sum(np.dot(I,Y**2))   / np.power(m00,2.0);
    n30 = np.sum(np.dot(X**3,I))   / np.power(m00,2.5);
    n03 = np.sum(np.dot(I,Y**3))   / np.power(m00,2.5);
    n21 = np.dot(X**2,np.dot(I,Y)) / np.power(m00,2.5);
    n12 = np.dot(X,np.dot(I,Y**2)) / np.power(m00,2.5);
    
    #M = np.array(M);
    #M = M.reshape((1,M.shape[0]))
    
    N = np.zeros(7);
    N[0] = n20 + n02;
    N[1] = (n20-n02)**2 + 4.0*n11**2;
    N[2] = (n30-3.0*n12)**2 + (n03-3.0*n21)**2;
    N[3] = (n30+n12)**2 + (n03+n21)**2;
    N[4] = (n30-3.0*n12)*(n30+n12)*((n30+n12)**2 -3.0*(n21+n03)**2)-(3.0*n21-n03)*(n21+n03)*(3.0*(n30+n12)**2-(n21+n03)**2);
    N[5] = (n20-n02)*((n30+n12)**2-(n21+n03)**2)+4.0*n11*(n30+n12)*(n03+n21);
    N[6] = (3.0*n21-n03)*(n30+n12)*((n30+n12)**2 -3.0*(n21+n03)**2) - (n30-3.0*n12)*(n21+n03)*(3.0*(n30+n12)**2 -(n21+n03)**2);
    
    return N.reshape((1,N.shape[0]));

X = scio.loadmat("mnist_all.mat");

B = 100;
x = None;
c = None;
for d in [0,1,2,3]:
    if x is None:
        x = X['test'+str(d)][:B];
        c = B*[0];
    else:
        x = np.r_[x,X['test'+str(d)][:B]];
        c = c + B*[d];
c = np.array(c);

y = hu_moments(x[0].reshape((28,28)));
for i in range(1,x.shape[0]):
    y = np.r_[y,hu_moments(x[i].reshape((28,28)))];

y_c = 1.0*y;
##for i in range(x.shape[0]):
    ##y_c[i] /= np.sum(np.abs(y_c[i]));
y_m = np.mean(y_c,axis=0);
for i in range(x.shape[0]):
    y_c[i] -= y_m;

K = np.zeros((x.shape[0],x.shape[0]));
for i in range(x.shape[0]):
    for j in range(i,x.shape[0]):
        K[i,j] = K[j,i] = np.log(1.0+np.exp(-np.dot(y_c[i],y_c[j])));

D = np.outer(np.diag(K),np.ones(x.shape[0])) + np.outer(np.ones(x.shape[0]),np.diag(K)) - 2.0*K;

J = np.eye(x.shape[0])-np.ones((x.shape[0],x.shape[0]))/x.shape[0];

B = -0.5*np.dot(J,np.dot(D,J));

E,V = np.linalg.eig(B);
Z = np.real(V[:,:2]);

plt.scatter(Z[:,0],Z[:,1],c=c);