import numpy as np 
from scipy import linalg
import math 

class PCCA_GC_Calculator:
    
    import math
    import numpy as np
    from scipy import linalg

    def __init__(self, X, Y_cause):
        """data for test whether the time series Y_cause causes X
        :param X: dim(X) = (N, dm) = (length of time series, dimension of X)
        :param Y_cause:
        """
        self.X = X
        self.Y = Y_cause
        self.X_t = X.T
        self.Y_t = Y_cause.T

    def calcSigmaHat(self, sigma, eta):
        return sigma + eta * np.identity(sigma.shape[0])

    def calcGrangerCausality(self, k, m, eta_xt=0.00001, eta_yt= 0.00001, eta_xtkm=0.00001):
        """
        :param k:
        :param m:
        :param eta_xt:
        :param eta_yt:
        :param eta_xtkm:
        :return:
        """
        N = self.X.shape[0]
        dim_X = self.X.shape[1]
        dim_Y = self.Y.shape[1]


        x_t = []
        y_t = []
        x_tk_m = []
        y_tk_m = []

        for t in range(k + m - 1, N):
            x_t.append(self.X[t])
            y_t.append(self.Y[t])

            cut_x = self.X[t - k - m + 1: t - k + 1]
            x_tk_m.append(np.ravel(cut_x[::-1]))         # reverse the array and make the array 1d array
            cut_y = self.Y[t - k - m + 1: t - k + 1]
            y_tk_m.append(np.ravel(cut_y[::-1]))         # reverse the array and make the array 1d array

        x_t = (np.array(x_t)).T
        y_t = (np.array(y_t)).T
        x_tk_m = (np.array(x_tk_m)).T
        y_tk_m = (np.array(y_tk_m)).T

        dim_x_t = x_t.shape[0]
        dim_y_t = y_t.shape[0]
        dim_x_tk_m = x_tk_m.shape[0]

        x = np.r_[x_t, y_t]
        y = np.r_[x_tk_m, y_tk_m]

        sigma = np.cov(m=x, y=y, rowvar=True)   # row of x and y represents a variable, and each column a single observation
        """ 
        sigma = ( sigma_xt_xt   sigma_xt_yt     sigma_xt_xtkm       sigma_xt_ytkm ) 
                ( sigma_yt_xt   sigma_yt_yt     sigma_yt_xtkm       sigma_yt_ytkm )
                ( sigma_xtkm_xt sigma_xtkm_yt   sigma_xtkm_xtkm     sigma_xtkm_ytkm )
                ( sigma_ytkm_xt sigma_ytkm_yt   sigma_ytkm_xtkm     sigma_ytkm_ytkm )
        """

        yt_start_idx = dim_x_t
        xtkm_start_idx = dim_x_t + dim_y_t
        ytkm_start_idx = dim_x_t + dim_y_t + dim_x_tk_m

        sigma_xt_xt = sigma[    0 : yt_start_idx, 0              : yt_start_idx]
        sigma_xt_xtkm = sigma[  0 : yt_start_idx, xtkm_start_idx : ytkm_start_idx]
        sigma_xt_ytkm = sigma[  0 : yt_start_idx, ytkm_start_idx :]

        sigma_xtkm_xt = sigma[  xtkm_start_idx : ytkm_start_idx, 0 : yt_start_idx]
        sigma_xtkm_xtkm = sigma[xtkm_start_idx : ytkm_start_idx, xtkm_start_idx : ytkm_start_idx]
        sigma_xtkm_ytkm = sigma[xtkm_start_idx : ytkm_start_idx, ytkm_start_idx : ]

        sigma_ytkm_xt = sigma[  ytkm_start_idx:, 0              : yt_start_idx]
        sigma_ytkm_xtkm = sigma[ytkm_start_idx:, xtkm_start_idx : ytkm_start_idx]
        sigma_ytkm_ytkm = sigma[ytkm_start_idx:, ytkm_start_idx : ]


        sigma_tilde_ytkm_xt_xtkm = sigma_ytkm_xt\
                                 - np.dot(np.dot(2 * sigma_ytkm_xtkm, np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))), sigma_xtkm_xt)\
                                 + np.dot(np.dot(np.dot(np.dot(sigma_ytkm_xtkm, np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))),sigma_xtkm_xtkm), np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))), sigma_xtkm_xt)

        sigma_tilde_xt_xt_xtkm = sigma_xt_xt \
                                   - np.dot(np.dot(2 * sigma_xt_xtkm, np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))), sigma_xtkm_xt) \
                                   + np.dot(np.dot(np.dot(np.dot(sigma_xt_xtkm, np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))),sigma_xtkm_xtkm), np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))), sigma_xtkm_xt)

        sigma_tilde_xt_ytkm_xtkm = sigma_xt_ytkm \
                                   - np.dot(np.dot(2 * sigma_xt_xtkm, np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))), sigma_xtkm_ytkm) \
                                   + np.dot(np.dot(np.dot(np.dot(sigma_xt_xtkm, np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))),sigma_xtkm_xtkm), np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))), sigma_xtkm_ytkm)

        sigma_tilde_ytkm_ytkm_xtkm = sigma_ytkm_ytkm \
                                   - np.dot(np.dot(2 * sigma_ytkm_xtkm, np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))), sigma_xtkm_ytkm) \
                                   + np.dot(np.dot(np.dot(np.dot(sigma_ytkm_xtkm, np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))),sigma_xtkm_xtkm), np.linalg.inv(self.calcSigmaHat(sigma=sigma_xtkm_xtkm, eta=eta_xtkm))), sigma_xtkm_ytkm)

        A = np.dot(np.dot(sigma_tilde_ytkm_xt_xtkm, np.linalg.inv(sigma_tilde_xt_xt_xtkm + eta_xt * np.identity(sigma_tilde_xt_xt_xtkm.shape[0]))), sigma_tilde_xt_ytkm_xtkm)
        B = sigma_tilde_ytkm_ytkm_xtkm + eta_yt * np.identity(sigma_tilde_ytkm_ytkm_xtkm.shape[0])

        eigenvalues = np.real(linalg.eig(a=A, b=B)[0])
        eigenvalue = np.max(eigenvalues)
        if eigenvalue > 1.0:
            eigenvalue = 0.9999
        Gyx = 0.5 * math.log(1 / (1 - eigenvalue), 2)

        return Gyx


def pcca_cause(img1, img2, 
                k=1, m=3, 
                eta_xt=5e-4, 
                eta_yt=5e-4,
                eta_xtkm=5e-4):
    
    """
    img1 : (M,N,T) array
    img2 : (M,N,T) array

    """
    Y = img1.copy() #- np.mean(img1)
    X = img2.copy() #- np.mean(img1)
    
    Y = Y.reshape(-1, Y.shape[-1]).T
    X = X.reshape(-1, X.shape[-1]).T
    
    calc_xy = PCCA_GC_Calculator(X=X, Y_cause=Y)
    # Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=1,
    #                                        eta_xt=1e-5, eta_yt=1e-5, eta_xtkm=1e-5) # delay lag=1 and order=1 # this is slow.... 
    Gy_to_x = calc_xy.calcGrangerCausality(k=k, m=m,
                                           eta_xt=eta_xt, 
                                           eta_yt=eta_yt, 
                                           eta_xtkm=eta_xtkm) # etas are very important 
    return Gy_to_x


def pcca_cause_pixel(img1, img2, 
                     k=1, 
                     m=3, 
                     eta_xt=5e-4, 
                     eta_yt=5e-4,
                     eta_xtkm=5e-4):

    import numpy as np 

    N = len(img1)
    corr_array = np.zeros((N,N)) 

    # iterate over every pixel comparison with the central... 

    

def pcca_cause_block(img1, img2,
                    block_size=3,
                     k=1, 
                     m=3, 
                     eta_xt=5e-4, 
                     eta_yt=5e-4,
                     eta_xtkm=5e-4):

    """
    breaks into a 3x3 block!. 
    """
    import numpy as np 

    corr_array = np.zeros((3,3))
    centre_ref = img1[1*block_size:2*block_size, 1*block_size:2*block_size].copy()


    top_left = img2[:block_size,:block_size].copy()
    top_center = img2[:block_size, block_size:2*block_size].copy()
    top_right = img2[:block_size, 2*block_size:3*block_size].copy()

    corr_array[0,0] = pcca_cause(centre_ref, top_left, 
                                    k=k, m=m, 
                                    eta_xt=eta_xt, 
                                    eta_yt=eta_yt,
                                    eta_xtkm=eta_xtkm)
    corr_array[0,1] = pcca_cause(centre_ref, top_center,
                                    k=k, m=m, 
                                    eta_xt=eta_xt, 
                                    eta_yt=eta_yt,
                                    eta_xtkm=eta_xtkm)
    corr_array[0,2] = pcca_cause(centre_ref, top_right,
                                    k=k, m=m, 
                                    eta_xt=eta_xt, 
                                    eta_yt=eta_yt,
                                    eta_xtkm=eta_xtkm)


    left = img2[1*block_size:2*block_size, :block_size].copy()
    center = img2[1*block_size:2*block_size, block_size:2*block_size].copy()
    right = img2[1*block_size:2*block_size, 2*block_size:3*block_size].copy()

    corr_array[1,0] = pcca_cause(centre_ref, left, 
                                            k=k, m=m, 
                                            eta_xt=eta_xt, 
                                            eta_yt=eta_yt,
                                            eta_xtkm=eta_xtkm)
    corr_array[1,1] = pcca_cause(centre_ref, center,
                                    k=k, m=m, 
                                    eta_xt=eta_xt, 
                                    eta_yt=eta_yt,
                                    eta_xtkm=eta_xtkm)
    corr_array[1,2] = pcca_cause(centre_ref, right,
                                    k=k, m=m, 
                                    eta_xt=eta_xt, 
                                    eta_yt=eta_yt,
                                    eta_xtkm=eta_xtkm)

    bottom_left = img2[2*block_size:3*block_size, :block_size].copy()
    bottom_center = img2[2*block_size:3*block_size, block_size:2*block_size].copy()
    bottom_right = img2[2*block_size:3*block_size, 2*block_size:3*block_size].copy()
    
    corr_array[2,0] = pcca_cause(centre_ref, bottom_left,
                                    k=k, m=m, 
                                    eta_xt=eta_xt, 
                                    eta_yt=eta_yt,
                                    eta_xtkm=eta_xtkm)
    corr_array[2,1] = pcca_cause(centre_ref, bottom_center,
                                    k=k, m=m, 
                                    eta_xt=eta_xt, 
                                    eta_yt=eta_yt,
                                    eta_xtkm=eta_xtkm)
    corr_array[2,2] = pcca_cause(centre_ref, bottom_right,
                                    k=k, m=m, 
                                    eta_xt=eta_xt, 
                                    eta_yt=eta_yt,
                                    eta_xtkm=eta_xtkm)
    corr_array = corr_array
    corr_array[np.isnan(corr_array)] = 0 

    return corr_array


def PCCA_causal_flow(imgseries1, imgseries2, 
                smooth_sigma=1, 
                winsize=3, 
                stride=3, 
                k=1, m=3, 
                eta_xt=5e-4, 
                eta_yt=5e-4,
                eta_xtkm=5e-4):

    import numpy as np 
    import scipy.ndimage as ndimage 
    from tqdm import tqdm 

    frame_a = imgseries1.copy()
    frame_b = imgseries2.copy()
    
    # prepad
    frame_a_ = np.pad(frame_a, [[winsize,winsize], [winsize,winsize], [0,0]], mode='constant', constant_values=0)
    frame_b_ = np.pad(frame_b, [[winsize,winsize], [winsize,winsize], [0,0]], mode='constant', constant_values=0)
    
    M,N = frame_a.shape[:2]
    # figure out the number of window iterations to take. 
    row_indices = np.arange(0, M-winsize, stride) + winsize # these are the central coordinates. 
    col_indices = np.arange(0, N-winsize, stride) + winsize 
    
    # now we can iterate and crop out. 
    out_vect = np.zeros((len(row_indices), len(col_indices), 2)) # out motion vectors
    xy_coords = np.zeros((len(row_indices), len(col_indices), 2)) # out xy coords. 

    row_ii = 0
    row_jj = 0
    
    for row_ii in tqdm(np.arange(len(row_indices[:]))):
        # for row_jj in tqdm(np.arange(len(col_indices[:]))): 
        for row_jj in np.arange(len(col_indices[:])): 

            # starting coordinates
            rr = row_indices[row_ii]#-winsize//2
            cc = col_indices[row_jj]#-winsize//2
            
            xy_coords[row_ii, row_jj,0] = cc - winsize//2
            xy_coords[row_ii, row_jj,1] = rr - winsize//2
            
            corr_array = np.zeros((3,3))
            
            # prev frame 
            centre_ref = frame_a_[rr:rr+winsize,cc:cc+winsize].copy() # the last axis is time!.
            
            # # next frame 
            top_left = frame_b_[rr-winsize:rr,cc-winsize:cc].copy()
            top_center = frame_b_[rr-winsize:rr,cc:cc+winsize].copy()
            top_right = frame_b_[rr-winsize:rr,cc+winsize:cc+2*winsize].copy()

            corr_array[0,0] = pcca_cause(centre_ref, top_left, 
                                            k=k, m=m, 
                                            eta_xt=eta_xt, 
                                            eta_yt=eta_yt,
                                            eta_xtkm=eta_xtkm)
            corr_array[0,1] = pcca_cause(centre_ref, top_center,
                                            k=k, m=m, 
                                            eta_xt=eta_xt, 
                                            eta_yt=eta_yt,
                                            eta_xtkm=eta_xtkm)
            corr_array[0,2] = pcca_cause(centre_ref, top_right,
                                            k=k, m=m, 
                                            eta_xt=eta_xt, 
                                            eta_yt=eta_yt,
                                            eta_xtkm=eta_xtkm)
    
            left = frame_b_[rr:rr+winsize,cc-winsize:cc].copy()
            center = frame_b_[rr:rr+winsize,cc:cc+winsize].copy()
            right = frame_b_[rr:rr+winsize,cc+winsize:cc+2*winsize].copy()
            
            corr_array[1,0] = pcca_cause(centre_ref, left, 
                                            k=k, m=m, 
                                            eta_xt=eta_xt, 
                                            eta_yt=eta_yt,
                                            eta_xtkm=eta_xtkm)
            corr_array[1,1] = pcca_cause(centre_ref, center,
                                            k=k, m=m, 
                                            eta_xt=eta_xt, 
                                            eta_yt=eta_yt,
                                            eta_xtkm=eta_xtkm)
            corr_array[1,2] = pcca_cause(centre_ref, right,
                                            k=k, m=m, 
                                            eta_xt=eta_xt, 
                                            eta_yt=eta_yt,
                                            eta_xtkm=eta_xtkm)

            bottom_left = frame_b_[rr+winsize:rr+2*winsize, cc-winsize:cc].copy()
            bottom_center = frame_b_[rr+winsize:rr+2*winsize, cc:cc+winsize].copy()
            bottom_right = frame_b_[rr+winsize:rr+2*winsize, cc+winsize:cc+2*winsize].copy()
            
            corr_array[2,0] = pcca_cause(centre_ref, bottom_left,
                                            k=k, m=m, 
                                            eta_xt=eta_xt, 
                                            eta_yt=eta_yt,
                                            eta_xtkm=eta_xtkm)
            corr_array[2,1] = pcca_cause(centre_ref, bottom_center,
                                            k=k, m=m, 
                                            eta_xt=eta_xt, 
                                            eta_yt=eta_yt,
                                            eta_xtkm=eta_xtkm)
            corr_array[2,2] = pcca_cause(centre_ref, bottom_right,
                                            k=k, m=m, 
                                            eta_xt=eta_xt, 
                                            eta_yt=eta_yt,
                                            eta_xtkm=eta_xtkm)
    
            corr_array[np.isnan(corr_array)] = 0 
            # corr_array[1,1] = 0 #np.nanmean(corr_array[corr_array>0])
            # corr_array = normxcorr2(centre_ref, center)
            mid = corr_array.shape[1]//2
            
            corr_x_direction = -corr_array[:, :mid].sum() + corr_array[:,mid+1:].sum()
            corr_y_direction = -corr_array[:mid].sum() + corr_array[mid+1:].sum()
            intensity = np.sum(corr_array) #* np.sqrt(corr_x_direction**2 + corr_y_direction**2)
            
            mean_vector = np.hstack([corr_y_direction, corr_x_direction])
            mean_vector = mean_vector * intensity
            
            out_vect[row_ii,row_jj,:] = mean_vector
            
            row_jj += 1
        row_ii +=1
            
    # # now we  apply smoothing to derive the flows!. ---> this is quite important!. 
    out_vect[...,0] = ndimage.gaussian_filter(out_vect[...,0], sigma=smooth_sigma)
    out_vect[...,1] = ndimage.gaussian_filter(out_vect[...,1], sigma=smooth_sigma)

    return xy_coords, out_vect 