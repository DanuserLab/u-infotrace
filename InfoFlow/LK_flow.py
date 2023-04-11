#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:22:15 2022

@author: felix
"""


def Linear_LK(X, alpha=1e-2, eps=1e-12): # this might actually be the same as DDC..... if we were to expand... the entries... 
    """
    http://www.ncoads.org/upload/202009/24/202009241429568694.pdf
        # do joint estimation. 
        for 2->1 
        
    # Linear LK is a covariance weighted ratio of the least mean squared weights!. 
        
    """

    import numpy as np 
    X_ = X.copy()
    # X_ = X_ - np.nanmean(X_, axis=-1)[...,None] # demean! 
    
    dX_ = X_[:,1:] - X_[:,:-1] # grab the differential. 
    X_ = X_[:,1:].copy()
    # T = X_.shape[-1] # number of timepoints, 

    X_ = X_.T
    dX_ = dX_.T
    
    # linear least squares solution . 
    dX_X = dX_.T.dot(X_)
    X_X = X_.T.dot(X_)
    
    # W = np.linalg.solve(X_X+reg*np.eye(len(X_X)), dX_X) # transpose... 
    W = dX_X.dot(np.linalg.inv(X_X +alpha*np.eye(len(X_X))))
    
    # return the least squares solution and X_X. the covariance! as the linear LK is a weighted multiplication!. 
    return W, X_X 
    # # estimates all the covariances... 
    # X_X = X_.T.dot(X_) / float(T) # this is covariance. # N x N signals. 
    # X_dX = X_.T.dot(dX_) / float(T)# this will give now... C_val->dval., where row to col. 
    
    # # create a couple of submatrices....
    # C_ij = X_X.copy() 
    # C_ii = np.tile(np.diag(X_X), (len(X_X),1)).T # this will broadcast all the rows! #Cjj is the same as Cii... 

    # C_j_di = X_dX.T.copy()
    # c_i_di = np.tile(np.diag(C_j_di), (len(C_j_di),1)).T

    # W = (C_ii*C_ij*C_j_di - C_ij**2*c_i_di) / (C_ii**2*C_ii - C_ii*C_ij**2 + 1e-12) # is this correct? 

    # return W 


def liang(y1, y2, npt=1, eps=1e-12):
    '''
    Estimate the Liang information transfer from series y2 to series y1 

    Parameters
    ----------

    y1, y2 : array
        Vectors of (real) numbers with identical length, no NaNs allowed

    npt : int  >=1
        Time advance in performing Euler forward differencing,
        e.g., 1, 2. Unless the series are generated with a highly chaotic deterministic system,
        npt=1 should be used

    Returns
    -------

    res : dict
        A dictionary of results including:
            T21 : float
                information flow from y2 to y1 (Note: not y1 -> y2!)
            tau21 : float
                the standardized information flow from y2 to y1
            Z : float
                the total information flow from y2 to y1
            dH1_star : float
                dH*/dt (Liang, 2016)
            dH1_noise : float
            
    See also
    --------

    pyleoclim.utils.causality.liang_causality : information flow estimated using the Liang algorithm
    pyleoclim.utils.causality.granger_causality : information flow estimated using the Granger algorithm    
    pyleoclim.utils.causality.signif_isopersist : significance test with AR(1) with same persistence
    pyleoclim.utils.causality.signif_isospec : significance test with surrogates with randomized phases
    
    References
    ----------

    Liang, X.S. (2013) The Liang-Kleeman Information Flow: Theory and
            Applications. Entropy, 15, 327-360, doi:10.3390/e15010327
    
    Liang, X.S. (2014) Unraveling the cause-effect relation between timeseries.
        Physical review, E 90, 052150
    
    Liang, X.S. (2015) Normalizing the causality between time series.
        Physical review, E 92, 022126
    
    Liang, X.S. (2016) Information flow and causality as rigorous notions ab initio.
        Physical review, E 94, 052201

    '''
    import numpy as np 

    dt=1
    nm = np.size(y1)

    grad1 = (y1[0+npt:] - y1[0:-npt]) / (npt)
    grad2 = (y2[0+npt:] - y2[0:-npt]) / (npt)

    y1 = y1[:-npt]
    y2 = y2[:-npt]

    N = nm - npt
    C = np.cov(y1, y2)
    detC = np.linalg.det(C)

    dC = np.ndarray((2, 2))
    dC[0, 0] = np.nansum((y1-np.nanmean(y1))*(grad1-np.nanmean(grad1)))
    dC[0, 1] = np.nansum((y1-np.nanmean(y1))*(grad2-np.nanmean(grad2)))
    dC[1, 0] = np.nansum((y2-np.nanmean(y2))*(grad1-np.nanmean(grad1)))
    dC[1, 1] = np.nansum((y2-np.nanmean(y2))*(grad2-np.nanmean(grad2)))

    dC /= N-1

    a11 = C[1, 1]*dC[0, 0] - C[0, 1]*dC[1, 0]
    a12 = -C[0, 1]*dC[0, 0] + C[0, 0]*dC[1, 0]

    a11 /= (detC+eps)
    a12 /= (detC+eps)

    # f1 = np.mean(grad1) - a11*np.mean(y1) - a12*np.mean(y2)
    # R1 = grad1 - (f1 + a11*y1 + a12*y2)
    # Q1 = np.sum(R1*R1)
    # b1 = np.sqrt(Q1*dt/N)

    # NI = np.ndarray((4, 4))
    # NI[0, 0] = N*dt/b1**2
    # NI[1, 1] = dt/b1**2*np.sum(y1*y1)
    # NI[2, 2] = dt/b1**2*np.sum(y2*y2)
    # NI[3, 3] = 3*dt/b1**4*np.sum(R1*R1) - N/b1**2
    # NI[0, 1] = dt/b1**2*np.sum(y1)
    # NI[0, 2] = dt/b1**2*np.sum(y2)
    # NI[0, 3] = 2*dt/b1**3*np.sum(R1)
    # NI[1, 2] = dt/b1**2*np.sum(y1*y2)
    # NI[1, 3] = 2*dt/b1**3*np.sum(R1*y1)
    # NI[2, 3] = 2*dt/b1**3*np.sum(R1*y2)

    # NI[1, 0] = NI[0, 1]
    # NI[2, 0] = NI[0, 2]
    # NI[2, 1] = NI[1, 2]
    # NI[3, 0] = NI[0, 3]
    # NI[3, 1] = NI[1, 3]
    # NI[3, 2] = NI[2, 3]

    # invNI = np.linalg.pinv(NI)
    # var_a12 = invNI[2, 2]
    T21 = C[0, 1]/C[0, 0] * (-C[1, 0]*dC[0, 0] + C[0, 0]*dC[1, 0]) / (detC+eps)
    # var_T21 = (C[0, 1]/C[0, 0])**2 * var_a12

    return T21#, var_T21 


# def linear_LK(X, p=1, eps=1e-12):

#     import numpy as np 
#     dX_ = (X[0,p:] - X[0,:-p])/ float(p)
#     X_ = X[:,p:].copy()
#     N = X_.shape[1]
    
#     print(X.shape)
#     # X_ = X_.T
#     # dX_ = dX_.T
#     dC = []
#     for k in np.arange(len(X)):
#         # %dC(1,1) = sum((x(:,1) - mean(x(:,1))) .* (dx1 - mean(dx1))); 
#         dC.append(np.sum((X_[k] - np.nanmean(X_[k])) * (dX_ - np.nanmean(dX_)))); 
#         # end
#     dC = np.array(dC)
#     dC = dC / (N-1);

#     print(dC.shape, dX_.shape)
#     C = np.cov(X_) # MxM matrix
#     print(C.shape, dC.shape)
    
#     ain = np.dot(np.linalg.pinv(C), dC)
#     print(ain.shape)
#     a12 = ain[1]
    
#     print(C)
#     print('ain shape', ain.shape)
    
#     T21 = C[0,1]/C[0,0] * a12
    
#     print(T21)
#     print(T21.shape)


def Linear_LK_cause(img1, img2, eps=1e-12, alpha=1e-2):

    import numpy as np 
    import pylab as plt 
    # compile all the timeseries
    Y_ = np.array([img1, 
                    img2])
    Y_ = Y_.reshape(len(Y_), -1, Y_.shape[-1])

    """
    Compute the diff covariance 
    """
    W_, C_ = Linear_LK(Y_.reshape(-1, Y_.shape[-1]), eps=eps)
    
    # W_ = W_.T.copy()
    N = Y_.shape[1] # this is the flattened over spatial windows. 
    N_rows = int(np.sqrt(N))

    # W_out = W_[1:,0].copy() # - W_[0,1:]
    W_out = W_[:N, N:].copy() # crop out the block and we need the influence... to only the central pixel!. 
    # W_out = W_[N:, :N].copy()
    
    # print(W_out.min(), W_out.max())
    corr_array = np.zeros((N_rows,N_rows))

    for ii in np.arange(N_rows):
        for jj in np.arange(N_rows):
            ind = ii*N_rows + jj
            # corr_array[ii,jj] = W_out[N//2, ind] # influence on the previous timepoint central pixel!. 
            
            #### this needs double checking!. ---> i think the indices are round!. 
            corr_array[ii,jj] = W_out[N//2, ind] * np.sqrt(C_[N//2,ind+N] / C_[N//2,N//2] + eps) # influence on the previous timepoint central pixel!.
            corr_array[ii,jj] = W_out[ind, ind] * np.sqrt(C_[ind,ind+N] / C_[ind,ind] + eps) # influence on the previous timepoint central pixel!.
    
    return corr_array


# def Linear_LK_cause(img1, img2, eps=1e-12, alpha=1e-2):

#     import numpy as np 
#     import pylab as plt 
    
#     N = img1.shape[0]

#     y1 = img1[N//2,N//2].copy()
#     y2 = img2.reshape(-1, img2.shape[-1]) # collapse space x time. 

    
#     # print(W_out.min(), W_out.max())
#     corr_array = np.zeros((N,N))

#     for ii in np.arange(N):
#         for jj in np.arange(N):
#             ind = ii*N + jj
#             corr_array[ii,jj] = liang(y1, y2[ind], npt=1) # influence on the previous timepoint central pixel!. 
    
#     return corr_array
    
