#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:22:15 2022

@author: s434626
"""

def differential_covariance(X, eps=1e-12, alpha=1e-3):
    """ uses ridge regularization 
    """
    import numpy as np 
    
    # standardize
    # X_ = (X - np.nanmean(X, axis=1)[:,None])  / ( np.nanstd(X, axis=1)[:,None] + eps )
    X_ = X.copy()
    
    # # differential 
    # X_pad = np.pad(X_, pad_width=[[0,0],[1,1]], mode='edge')
    # dX_ = (X_pad[:,2:] - X_pad[:,:-2]) / 2.
    # # dX_ = np.gradient(X_, axis=1)
    dX_ = X_[:,1:] - X_[:,:-1]
    X_ = X_[:,1:].copy()
    # print(X_.shape,dX_.shape)
    X_ = X_.T
    dX_ = dX_.T
    
    # linear least squares solution . 
    dX_X = dX_.T.dot(X_)
    X_X = X_.T.dot(X_)
    
    # W = np.linalg.solve(X_X+reg*np.eye(len(X_X)), dX_X) # transpose... 
    W = dX_X.dot(np.linalg.inv(X_X +alpha*np.eye(len(X_X))))
    
    return W 


def DDC_cause(img1, img2, eps=1e-12, alpha=1e-2):

    import numpy as np 
    import pylab as plt 
    
    # compile all the timeseries
    Y_ = np.array([img1, 
                   img2])
    Y_ = Y_.reshape(len(Y_), -1, Y_.shape[-1])

    """
    Compute the diff covariance 
    """
    W_ = differential_covariance(Y_.reshape(-1, Y_.shape[-1]), eps=eps, alpha=alpha)
    
    # W_ = W_.T.copy()
    N = Y_.shape[1] # this is the flattened over spatial windows. 
    N_rows = int(np.sqrt(N))

    # W_out = W_[1:,0].copy() # - W_[0,1:]
    W_out = W_[:N, N:].copy()
    corr_array = np.zeros((N_rows,N_rows))

    for ii in np.arange(N_rows):
        for jj in np.arange(N_rows):
            ind = ii*N_rows + jj
            corr_array[ii,jj] = W_out[ind,ind]
    
    return corr_array
    
