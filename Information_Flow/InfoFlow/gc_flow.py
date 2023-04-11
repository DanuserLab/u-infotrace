#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:22:15 2022

@author: s205272
"""

def GC_full_reduced_separate_regress_individual(img1, img2, lag=1, alpha=.1):
        
        from sklearn.linear_model import Ridge
        import numpy as np 
            
        # initialise 
        """
        Reduced regression 
        """
        # reduced model 
        clf = Ridge(alpha=alpha)

        Y = (img1.reshape(-1,img1.shape[-1]).T)[lag:]
        X = []
        for ll in range(1,lag+1):
            X_ = (img1.reshape(-1,img1.shape[-1]).T)[lag-ll:-ll]
            X.append(X_)
        X = np.hstack(X)
        
        clf.fit(X,Y)
        
        # logL = np.prod(np.linalg.slogdet(np.cov(Y - clf.predict(X))))
        logL = np.log(np.var(Y - clf.predict(X), axis=0)) # .mean())
        
        """
        Full Regression
        """
        # full model 
        clf_full = Ridge(alpha=alpha)
        X_full = []
        # for ll in range(2,lag+1):
        #     X_ = (img1.reshape(-1,img1.shape[-1]).T)[lag-ll:-ll]
        #     X_full.append(X_)
        for ll in range(1,lag+1):
            X_ = (img2.reshape(-1,img2.shape[-1]).T)[lag-ll:-ll]
            X_full.append(X_)
        X_full.append((img2.reshape(-1,img2.shape[-1]).T)[lag:])
        # X_full.append((img2.reshape(-1,img2.shape[-1]).T)[lag-1:-1])
        X_full = np.hstack(X_full) # n_time x n_variables.
        
        clf_full.fit(X_full, Y)
        
        # logF = np.prod(np.linalg.slogdet(np.cov(Y - clf_full.predict(X_full))))
        logF = np.log(np.var(Y - clf_full.predict(X_full), axis=0)) #.mean())
        
        
        # get the difference!. # not a pval ... but a magnitude. 
        return logF - logL 
    
if __name__=="__main__":

    # import pyinform
    import numpy as np
    import scipy.io
    #import random
    # from pyinform import transfer_entropy 
    from scipy.ndimage import gaussian_filter
    from matplotlib import pyplot as plt
    # from PyIF import te_compute as te
    # import te_compute as te
    ## import matCellRatioT
    from skimage.transform import rescale, resize, downscale_local_mean
    from skimage import color
    from tqdm import tqdm 
    import skimage.transform as sktform
    import skimage.util as skutil 
    import scipy.ndimage as ndimage 
    
    #def rgb2gray(rgb):
    
    #   r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    #    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    #    return gray
    myVid = read_video_cv2(r'821-10_l.mov') #works ! # weird ? 
    # myVid = read_video_cv2(r'9-19_l.mov') #works !
    # myVid =  read_video_cv2(r'001-0436.avi') # works!
    # myVid = read_video_cv2(r'3687-18_70.mov')
    # myVid = read_video_cv2(r'620-72_l.mov')
    # myVid = read_video_cv2(r'637-147_l.mov')
    # myVid = read_video_cv2(r'341-46_l.mov')
    # myVid = read_video_cv2(r'7399-1_70.mov')
    # myVid = read_video_cv2(r'2082-3_70.mov')
    # myVid = read_video_cv2(r'1174-6_70.mov')
    # myVid = read_video_cv2(r'879-38_l.mov')
    
    # myVid = read_video_cv2(r'965-126_l.mov')
    # myVid = read_video_cv2(r'621-3_l.mov')
    
    # import imageio
    # # import numpy as np
    # vid = imageio.get_reader(r'7399-1_70.mov',  'ffmpeg')
    # myVid = np.array([im for im in vid.iter_data()], dtype=np.uint8)
    myVid = color.rgb2gray(myVid)
    # 
    myVid2 = sktform.resize(myVid, output_shape=(myVid.shape[0], myVid.shape[1]//2, myVid.shape[2]//2), preserve_range=True)
    myVid4 = sktform.resize(myVid, output_shape=(myVid.shape[0], myVid.shape[1]//4, myVid.shape[2]//4), preserve_range=True)
    myVid8 = sktform.resize(myVid, output_shape=(myVid.shape[0], myVid.shape[1]//8, myVid.shape[2]//8), preserve_range=True)
    

# =============================================================================
#     1. extract all windowed - flat
# =============================================================================
    
    GC_vectors = causal_flow(myVid, winsize=3, lag=5, k=1, alpha=1)
    GC_vectors2 = causal_flow(myVid2, winsize=3, lag=5, k=1, alpha=1)
    GC_vectors4 = causal_flow(myVid4, winsize=3, lag=5, k=1, alpha=1)
    GC_vectors8 = causal_flow(myVid8, winsize=3, lag=5, k=1, alpha=1)
    
    GC_vectors2_resize = np.dstack([sktform.resize(GC_vectors2[...,ch], output_shape=myVid.shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
    GC_vectors4_resize = np.dstack([sktform.resize(GC_vectors4[...,ch], output_shape=myVid.shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
    GC_vectors8_resize = np.dstack([sktform.resize(GC_vectors8[...,ch], output_shape=myVid.shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
    
    # what is the best way to combine? 
    # GC_vectors_combine = 1./4*(GC_vectors + GC_vectors2_resize + GC_vectors4_resize + GC_vectors8_resize)
    # GC_vectors_combine = 1*GC_vectors + 2*GC_vectors2_resize +4* GC_vectors4_resize + 8*GC_vectors8_resize
    # GC_vectors_combine = GC_vectors_combine / ( 1 + 2 + 4 + 8)
    
# =============================================================================
#     2. is this really the best way to combine ? 
# =============================================================================
    GC_vectors_combine = np.nanmean( np.array([GC_vectors, GC_vectors2_resize, GC_vectors4_resize, GC_vectors8_resize]), axis=0)
    
    xy_coords = np.indices(myVid.shape[1:]); xy_coords=xy_coords.transpose(1,2,0)
    xy_coords = xy_coords[...,::-1]
    
    sampling = 8
    plt.figure(figsize=(15,15))
    plt.imshow(myVid[1])
    plt.quiver(xy_coords[::sampling,::sampling,0], 
                xy_coords[::sampling,::sampling,1], 
                GC_vectors[::sampling,::sampling,1],  # x 
                -GC_vectors[::sampling,::sampling,0]) # y 
    plt.show()
    
    
    plt.figure(figsize=(15,15))
    plt.title('2x')
    plt.imshow(myVid[1])
    plt.quiver(xy_coords[::sampling,::sampling,0], 
                xy_coords[::sampling,::sampling,1], 
                GC_vectors2_resize[::sampling,::sampling,1],  # x 
                -GC_vectors2_resize[::sampling,::sampling,0]) # y 
    plt.show()
    
    
    plt.figure(figsize=(15,15))
    plt.title('4x')
    plt.imshow(myVid[1])
    plt.quiver(xy_coords[::sampling,::sampling,0], 
                xy_coords[::sampling,::sampling,1], 
                GC_vectors4_resize[::sampling,::sampling,1],  # x 
                -GC_vectors4_resize[::sampling,::sampling,0]) # y 
    plt.show()
    
    plt.figure(figsize=(15,15))
    plt.title('8x')
    plt.imshow(myVid[1])
    plt.quiver(xy_coords[::sampling,::sampling,0], 
                xy_coords[::sampling,::sampling,1], 
                GC_vectors8_resize[::sampling,::sampling,1],  # x 
                -GC_vectors8_resize[::sampling,::sampling,0]) # y 
    plt.show()
    
    
    plt.figure(figsize=(15,15))
    plt.title('Combine')
    plt.imshow(myVid[1])
    plt.quiver(xy_coords[::sampling,::sampling,0], 
                xy_coords[::sampling,::sampling,1], 
                GC_vectors_combine[::sampling,::sampling,1],  # x 
                -GC_vectors_combine[::sampling,::sampling,0]) # y 
    plt.show()
    
    # plt.figure(figsize=(15,15))
    # # plt.imshow(myVid[1])
    # plt.imshow(frame_a_[...,0])
    # plt.plot(x.ravel(), 
    #          y.ravel(), 'k.')
    # plt.show()
    
    
#     # # myVid = sktform.resize(myVid, output_shape=(myVid.shape[0], myVid.shape[1], myVid.shape[2]), pre serve_range=True)

#     # myVid = (myVid - np.mean(myVid)) / np.std(myVid)
#     # test = sliding_window_array(myVid[0], window_size = 64, overlap = 32)
#     # test_2 = moving_window_array(myVid[0], window_size = 64, overlap = 32)
#     """
#     Determine the padding based on winsize + striding. # winsize should determine this.
#     """
#     # winsize = 5 # take only normal sizes # block may not be good for this ... 
#     winsize = 3
#     stride = winsize # this is fine. 
    
#     # added settings to allow assessment of length of video. ---> this works!. wow.! 
#     start = 0
#     end = start + len(myVid)
    
#     frame_a = myVid[start:end].transpose(1,2,0)
#     frame_b = myVid[start:end].transpose(1,2,0)
    
#     # frame_a = frame_a.transpose(1,2,0)
#     # frame_b = frame_b.transpose(1,2,0)
#     frame_a_ = np.pad(frame_a, [[winsize,winsize], [winsize,winsize], [0,0]], mode='constant', constant_values=0)
#     frame_b_ = np.pad(frame_b, [[winsize,winsize], [winsize,winsize], [0,0]], mode='constant', constant_values=0)
#     # frame_a_ = np.pad(frame_a, [[winsize,winsize], [winsize,winsize], [0,0]], mode='edge')
#     # frame_b_ = np.pad(frame_b, [[winsize,winsize], [winsize,winsize], [0,0]], mode='edge')
    
#     # frame_a_ = np.pad(frame_a, [[winsize,winsize], [winsize,winsize]], mode='reflect')
#     # frame_b_ = np.pad(frame_b, [[winsize,winsize], [winsize,winsize]], mode='edge')
    
#     M,N = frame_a.shape[:2]
#     # figure out the number of window iterations to take. 
#     row_indices = np.arange(0, M-winsize, stride) + winsize # these are the central coordinates. 
#     col_indices = np.arange(0, N-winsize, stride) + winsize 
    
#     # now we can iterate and crop out. 
#     out_vect = np.zeros((len(row_indices), len(col_indices), 2))
#     xy_coords = np.zeros((len(row_indices), len(col_indices), 2))
    
#     def xcorr2(A,B, norm=True):
#         from scipy.signal import correlate2d
#         if norm:
#             # A_ = (A-np.nanmean(A, axis=1)[None,:] / (np.nanstd(A, axis=1))[None,:])
#             # B_ = (B-np.nanmean(B, axis=1)[None,:] / (np.nanstd(B, axis=1))[None,:])
#             A_ = (A-np.nanmean(A)) / (np.nanstd(A)*np.prod(A.shape[:2]))
#             B_ = (B-np.nanmean(B)) / (np.nanstd(B))
#         else:
#             A_ = A.copy()
#             B_ = B.copy()
            
#         # compute the dot product. 
#         return correlate2d(A_, B_, mode='same', boundary='fill')
    
    
#     def normxcorr2(A,B, norm=True):
#         from scipy.signal import correlate2d
#         from skimage.feature import match_template
        
#         # A_ = A-np.nanmean(A)
#         # B_ = B-np.nanmean(B)
        
#         # x = match_template(A,B, pad_input=True)
#         # return x 
#         # # return np.max(x)
#         # # return x[A.shape[0]//2, A.shape[1]//2]
#         # # return float(correlate2d(A-A.mean(), B-B.mean(), mode='valid'))
#         # # A_ = (A-np.nanmean(A)) / (np.nanstd(A)*np.prod(A.shape[:2]))
#         # # # B_ = (B-np.nanmean(B)) / (np.nanstd(B))
#         A_ = A-np.nanmean(A)
#         B_ = B-np.nanmean(B)
        
#         # # return float(np.max(correlate2d(A_,B_, mode='same')))
#         return correlate2d(A_,B_, mode='same')
#         # # return A_.ravel().dot(B_.ravel())
    
#     """
#     Directionality is weird? 
#     """
#     # is this sliding correct? 
#     # def granger_naive(img1, img2):
#     #     from CausalCalculator import CausalCalculator
#     #     ## zero_pad # we scan the 1st?
#     #     T, m, n = img2.shape # we need some time T!. 
#     #     img1_pad = np.pad(img1,[[0,0], [m-1,m-1],[n-1,n-1]], mode='edge') # do the padding
#     #     # img1_pad = np.pad(img1,[[0,0], [m-1,m-1],[n-1,n-1]], mode='reflect')#, constant_values=0)
#     #     out = np.zeros((m+n-1, m+n-1)) # full # granger causal intensities!. 
#     #     # out2 = np.zeros((m+n-1, m+n-1)) # full
#     #     """
#     #     we can speed this up by extracting all windows... and running list comprehensin ? 
#     #     """
#     #     M,N = out.shape
#     #     # for ii in tqdm(np.arange(M)):
#     #     for ii in np.arange(0,M):
#     #         for jj in np.arange(0,N):
#     #             Y = img1_pad[:,ii:ii+m,
#     #                            jj:jj+n].copy()
#     #             X = img2.copy()
#     #             # Y = Y.reshape(Y.shape[0], -1)
#     #             # X = X.reshape(X.shape[0], -1)
#     #             Y = Y.reshape(-1, Y.shape[-1]).T
#     #             X = X.reshape(-1, X.shape[-1]).T
#     #             X = X - X.mean()
#     #             X = Y - Y.mean()
#     #             calc_xy = CausalCalculator(X=Y, Y_cause=X)
#     #             Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=1) # delay lag=1 and order=1 # this is slow.... 
#     #             out[ii,jj] = Gy_to_x # scalar. 
#     #     return out
    
#     def granger_naive(img1, img2):
#         from CausalCalculator import CausalCalculator
#         ## zero_pad # we scan the 1st?
#         m, n, T = img2.shape # we need some time T!. 
#         img1_pad = np.pad(img1,[[m-1,m-1],[n-1,n-1], [0,0]], mode='constant', constant_values=0) # do the padding
#         # img1_pad = np.pad(img1,[[m-1,m-1],[n-1,n-1], [0,0]], mode='edge') # do the padding
#         # img1_pad = np.pad(img1,[[0,0], [m-1,m-1],[n-1,n-1]], mode='reflect')#, constant_values=0)
#         out = np.zeros((m+n-1, m+n-1)) # full # granger causal intensities!. 
#         # out2 = np.zeros((m+n-1, m+n-1)) # full
#         """
#         we can speed this up by extracting all windows... and running list comprehensin ? 
#         """
#         M,N = out.shape
#         # for ii in tqdm(np.arange(M)):
#         for ii in np.arange(0,M):
#             for jj in np.arange(0,N):
#                 Y = img1_pad[ii:ii+m,
#                              jj:jj+n,:].copy()
#                 X = img2.copy()
#                 # print(X.shape, Y.shape)
#                 # Y = Y.reshape(Y.shape[0], -1)
#                 # X = X.reshape(X.shape[0], -1)
#                 Y = Y.reshape(-1, Y.shape[-1]).T
#                 X = X.reshape(-1, X.shape[-1]).T
#                 # print(X.shape, Y.shape)
#                 X = X - X.mean()
#                 X = Y - Y.mean()
                
#                 # its like on or off.... 
#                 calc_xy = CausalCalculator(X=X, Y_cause=Y) # for some reason .... no magnitude.... 
#                 Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=1) # delay lag=1 and order=1 # this is slow.... 
#                 out[ii,jj] = Gy_to_x # scalar. 
                
#         # print(out.shape)
#         return out 
#         # return out[out.shape[0]//2-m//2:out.shape[0]//2-m//2+m, out.shape[1]//2-n//2:out.shape[1]//2-n//2+n]
    
#     def granger_naive2(img1, img2):
#         from CausalCalculator import CausalCalculator
#         ## zero_pad # we scan the 1st?
#         m1, n1, T1 = img1.shape
#         m2, n2, T2 = img2.shape # we need some time T!. 
        
#         shifts_m = m2-m1+1
#         shifts_n = n2-n1+1
        
#         out = np.zeros((shifts_m, shifts_n)) # full # granger causal intensities!. 
#         # out2 = np.zeros((m+n-1, m+n-1)) # full
#         """
#         we can speed this up by extracting all windows... and running list comprehensin ? 
#         """
#         M,N = out.shape
#         # print(M,N)
#         # for ii in tqdm(np.arange(M)):
#         for ii in np.arange(0,M):
#             for jj in np.arange(0,N):
#                 Y = img2[ii:ii+m1,
#                          jj:jj+n1,:].copy()
#                 X = img1.copy()
#                 # print(X.shape, Y.shape)
#                 # Y = Y.reshape(Y.shape[0], -1)
#                 # X = X.reshape(X.shape[0], -1)
#                 Y = Y.reshape(-1, Y.shape[-1]).T
#                 X = X.reshape(-1, X.shape[-1]).T
#                 X = X - X.mean()
#                 X = Y - Y.mean()
#                 calc_xy = CausalCalculator(X=X, Y_cause=Y)
#                 Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=1) # delay lag=1 and order=1 # this is slow.... 
#                 out[ii,jj] = Gy_to_x # scalar. 
#         return out
    
#     def pcca_cause(img1, img2):
        
#         from CausalCalculator import CausalCalculator
#         Y = img1.copy() #- np.mean(img1)
#         X = img2.copy() #- np.mean(img1)
        
#         Y = Y.reshape(-1, Y.shape[-1]).T
#         X = X.reshape(-1, X.shape[-1]).T
#         # print(X.shape, Y.shape)
#         # X = X - X.mean(axis=1)[:,None]
#         # X = Y - Y.mean(axis=1)[:,None]
#         # X = (X - X.mean())/(X.std()+1e-8) 
#         # Y = (Y - Y.mean())/(Y.std()+1e-8) 
#         # print(X.shape)
#         # print(Y.shape)
#         calc_xy = CausalCalculator(X=X, Y_cause=Y)
#         # Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=1,
#         #                                        eta_xt=1e-5, eta_yt=1e-5, eta_xtkm=1e-5) # delay lag=1 and order=1 # this is slow.... 
#         Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=3,
#                                                eta_xt=5e-4, 
#                                                eta_yt=5e-4, 
#                                                eta_xtkm=5e-4) # etas are very important 
#         return Gy_to_x
    
    
#     def TE_cause(img1, img2):
        
#         import te_compute as te
#         from copent import transent
        
#         Y = img1.copy() #- np.mean(img1)
#         X = img2.copy() #- np.mean(img1)
        
#         # xs = np.mean()
#         # myTE = te.te_compute(xs, ys, 1, 1)
        
#         Y = Y.reshape(-1, Y.shape[-1]).T
#         X = X.reshape(-1, X.shape[-1]).T
        
#         # print(X.shape, Y.shape)
#         xs = np.nanmean(X, axis=-1)
#         ys = np.nanmean(Y, axis=-1)
#         # print(xs.shape, ys.shape)
        
#         # # is the order wrong? 
#         TE_1 = te.te_compute(ys, #-np.nanmean(ys), 
#                             xs,#-np.nanmean(xs), 
#                             5, 3) # number of nearest neighbors needs to increase
#         TE_2 = te.te_compute(xs, #-np.nanmean(ys), 
#                             ys,#-np.nanmean(xs), 
#                             5, 3) # number of nearest neighbors needs to increase
        
#         TE = np.maximum(TE_1,TE_2) - np.minimum(TE_1,TE_2)
#         # TE = transent(ys, xs, lag = 5, k = 3, dtype = 2, mode = 1)
#         # # print(X.shape, Y.shape)
#         # # X = X - X.mean(axis=1)[:,None]
#         # # X = Y - Y.mean(axis=1)[:,None]
#         # # X = (X - X.mean())/(X.std()+1e-8) 
#         # # Y = (Y - Y.mean())/(Y.std()+1e-8) 
#         # # print(X.shape)
#         # # print(Y.shape)
#         # calc_xy = CausalCalculator(X=X, Y_cause=Y)
#         # # Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=1,
#         # #                                        eta_xt=1e-5, eta_yt=1e-5, eta_xtkm=1e-5) # delay lag=1 and order=1 # this is slow.... 
#         # Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=3,
#         #                                        eta_xt=5e-4, 
#         #                                        eta_yt=5e-4, 
#         #                                        eta_xtkm=5e-4) # etas are very important 
#         return TE
    
    
#     def PDF_cause_hack(img1, img2, p=3):
        
#         from pdc_dtf import mvar_fit, PDC
        
#         Y = img1.copy() #- np.mean(img1)
#         X = img2.copy() #- np.mean(img1)
        
#         # xs = np.mean()
#         # myTE = te.te_compute(xs, ys, 1, 1)
        
#         Y = Y.reshape(-1, Y.shape[-1])
#         X = X.reshape(-1, X.shape[-1])
#         N = len(Y)
#         # compute all the causalities. 
        
#         Y_ = np.vstack([X,Y])
#         mu = np.mean(Y_, axis=1)
#         X_ = Y_ - mu[:, None]
        
#         A_est, sigma = mvar_fit(X_, p)    
#         sigma = np.diag(sigma)  # DTF + PDC support diagonal noise
#         # sigma = None)
    
#         # compute PDC
#         # print(A_est.shape)
#         P, freqs = PDC(A_est, sigma)
        
#         # get the causalities of the block between X to Y.... 
#         P_xy = P[:, -N:, :N].copy() # this should be X -> Y 
#         # print(P_xy.shape)
#         P_xy = np.max(P_xy, axis=0) # maximum over all frequencies.... 
#         # print(P_xy.shape)
#         # we need to subtract this from some base... 
#         # return np.log(np.linalg.det(P_xy))
#         return np.prod(np.linalg.slogdet(P_xy))
    
    
#     def TE_pyinform_cause(img1, img2):
        
#         # import te_compute as te
#         from pyinform.transferentropy import transfer_entropy
        
#         Y = img1.copy() #- np.mean(img1)
#         X = img2.copy() #- np.mean(img1)
        
#         # xs = np.mean()
#         # myTE = te.te_compute(xs, ys, 1, 1)
        
#         Y = Y.reshape(-1, Y.shape[-1]).T
#         X = X.reshape(-1, X.shape[-1]).T
        
#         # # print(X.shape, Y.shape)
#         xs = np.nanmean(X, axis=-1)
#         ys = np.nanmean(Y, axis=-1)
        
#         # xs = np.nanmax(X, axis=-1)
#         # ys = np.nanmax(Y, axis=-1)
#         # print(xs.shape, ys.shape)
        
#         # is the order wrong? 
#         TE = transfer_entropy(ys, 
#                               xs, k=1) # number of nearest neighbors needs to increase
#         # # print(X.shape, Y.shape)
#         # # X = X - X.mean(axis=1)[:,None]
#         # # X = Y - Y.mean(axis=1)[:,None]
#         # # X = (X - X.mean())/(X.std()+1e-8) 
#         # # Y = (Y - Y.mean())/(Y.std()+1e-8) 
#         # # print(X.shape)
#         # # print(Y.shape)
#         # calc_xy = CausalCalculator(X=X, Y_cause=Y)
#         # # Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=1,
#         # #                                        eta_xt=1e-5, eta_yt=1e-5, eta_xtkm=1e-5) # delay lag=1 and order=1 # this is slow.... 
#         # Gy_to_x = calc_xy.calcGrangerCausality(k=1, m=3,
#         #                                        eta_xt=5e-4, 
#         #                                        eta_yt=5e-4, 
#         #                                        eta_xtkm=5e-4) # etas are very important 
#         return TE
    
    
#     def differential_covariance(X, eps=1e-12, reg=1e-3):
    
#         import numpy as np 
        
#         # standardize
#         X_ = (X - np.nanmean(X, axis=1)[:,None]) #/ ( np.nanstd(X, axis=1)[:,None] + eps )
#         # X_ = X.copy()
        
#         # # differential 
#         # X_pad = np.pad(X_, pad_width=[[0,0],[1,1]], mode='edge')
#         # dX_ = (X_pad[:,2:] - X_pad[:,:-2]) / 2.
#         # # dX_ = np.gradient(X_, axis=1)
#         dX_ = X_[:,1:] - X_[:,:-1]
#         X_ = X_[:,1:].copy()
#         # print(X_.shape,dX_.shape)
#         X_ = X_.T
#         dX_ = dX_.T
        
#         # linear least squares solution . 
#         dX_X = dX_.T.dot(X_)
#         X_X = X_.T.dot(X_)
        
#         # W = np.linalg.solve(X_X, dX_X) # transpose... 
#         W = dX_X.dot(np.linalg.inv(X_X+reg*np.eye(len(X_X))))
        
#         return W 
    
    
#     def GC_full_reduced_separate_regress(img1, img2, lag=1, alpha=.1):
        
#         from sklearn.linear_model import Ridge
#         import numpy as np 
            
#         # initialise 
        
#         """
#         Reduced regression 
#         """
#         # reduced model 
#         clf = Ridge(alpha=alpha)

#         Y = (img1.reshape(-1,centre_ref.shape[-1]).T)[lag:]
#         X = []
#         for ll in range(1,lag+1):
#             X_ = (img1.reshape(-1,img1.shape[-1]).T)[lag-ll:-ll]
#             X.append(X_)
#         X = np.hstack(X)
        
#         clf.fit(X,Y)
        
#         # logL = np.prod(np.linalg.slogdet(np.cov(Y - clf.predict(X))))
#         logL = np.log(np.var(Y - clf.predict(X), axis=0).mean())
        
#         """
#         Full Regression
#         """
#         # full model 
#         clf_full = Ridge(alpha=alpha)
#         X_full = []
#         for ll in range(1,lag+1):
#             X_ = (img1.reshape(-1,img1.shape[-1]).T)[lag-ll:-ll]
#             X_full.append(X_)
#         for ll in range(1,lag+1):
#             X_ = (img2.reshape(-1,img2.shape[-1]).T)[lag-ll:-ll]
#             X_full.append(X_)
#         X_full.append((img2.reshape(-1,img2.shape[-1]).T)[lag:])
#         X_full = np.hstack(X_full) # n_time x n_variables.
        
#         clf_full.fit(X_full, Y)
        
#         # logF = np.prod(np.linalg.slogdet(np.cov(Y - clf_full.predict(X_full))))
#         logF = np.log(np.var(Y - clf_full.predict(X_full), axis=0).mean())
        
        
#         # get the difference!. # not a pval ... but a magnitude. 
#         return logF - logL 
    
    
#     def GC_full_reduced_separate_regress_individual(img1, img2, lag=1, alpha=.1):
        
#         from sklearn.linear_model import Ridge
#         import numpy as np 
            
#         # initialise 
#         """
#         Reduced regression 
#         """
#         # reduced model 
#         clf = Ridge(alpha=alpha)

#         Y = (img1.reshape(-1,centre_ref.shape[-1]).T)[lag:]
#         X = []
#         for ll in range(1,lag+1):
#             X_ = (img1.reshape(-1,img1.shape[-1]).T)[lag-ll:-ll]
#             X.append(X_)
#         X = np.hstack(X)
        
#         clf.fit(X,Y)
        
#         # logL = np.prod(np.linalg.slogdet(np.cov(Y - clf.predict(X))))
#         logL = np.log(np.var(Y - clf.predict(X), axis=0)) # .mean())
        
#         """
#         Full Regression
#         """
#         # full model 
#         clf_full = Ridge(alpha=alpha)
#         X_full = []
#         for ll in range(1,lag+1):
#             X_ = (img1.reshape(-1,img1.shape[-1]).T)[lag-ll:-ll]
#             X_full.append(X_)
#         for ll in range(1,lag+1):
#             X_ = (img2.reshape(-1,img2.shape[-1]).T)[lag-ll:-ll]
#             X_full.append(X_)
#         X_full.append((img2.reshape(-1,img2.shape[-1]).T)[lag:])
#         X_full = np.hstack(X_full) # n_time x n_variables.
        
#         clf_full.fit(X_full, Y)
        
#         # logF = np.prod(np.linalg.slogdet(np.cov(Y - clf_full.predict(X_full))))
#         logF = np.log(np.var(Y - clf_full.predict(X_full), axis=0)) #.mean())
        
        
#         # get the difference!. # not a pval ... but a magnitude. 
#         return logF - logL 
        

#     import scipy.stats as spstats
#     from pdc_dtf import mvar_fit, PDC
#     row_ii = 0
#     row_jj = 0
    
#     for row_ii in tqdm(np.arange(len(row_indices[:]))):
#         # for row_jj in tqdm(np.arange(len(col_indices[:]))): 
#         for row_jj in np.arange(len(col_indices[:])): 
#             """
#             need to evaluate an actual convolution.... 
#             """
            
#             # starting coordinates
#             rr = row_indices[row_ii]#-winsize//2
#             cc = col_indices[row_jj]#-winsize//2
            
#             xy_coords[row_ii, row_jj,0] = cc - winsize//2
#             xy_coords[row_ii, row_jj,1] = rr - winsize//2
            
#             corr_array = np.zeros((3,3))
#             # # corr_array_vects = np.dstack([[[-1,-1,-1],
#             # #                               [0,0,0],
#             # #                               [1,1,1]], [[-1,0,1],
#             # #                                           [-1,0,1],
#             # #                                           [-1,0,1]]])
#             # # corr_array_vects = corr_array_vects/(np.linalg.norm(corr_array_vects, axis=-1)[...,None]+1e-8)
            
#             # last frame 
#             centre_ref = frame_a_[rr:rr+winsize,cc:cc+winsize].copy() # the last axis is time!.
            
#             # # next frame 
#             top_left = frame_b_[rr-winsize:rr,cc-winsize:cc].copy()
#             top_center = frame_b_[rr-winsize:rr,cc:cc+winsize].copy()
#             top_right = frame_b_[rr-winsize:rr,cc+winsize:cc+2*winsize].copy()
    
#             # # # corr_array[0,0] = spstats.pearsonr(centre_ref.ravel(), top_left.ravel())[0]
#             # # # corr_array[0,1] = spstats.pearsonr(centre_ref.ravel(), top_center.ravel())[0]
#             # # # corr_array[0,2] = spstats.pearsonr(centre_ref.ravel(), top_right.ravel())[0]
#             # # corr_array[0,0] = GC_full_reduced_separate_regress(top_left, centre_ref, lag=5, alpha=.1) #(centre_ref, top_left)
#             # # corr_array[0,1] = GC_full_reduced_separate_regress(top_center, centre_ref, lag=5, alpha=.1)#(centre_ref, top_center)
#             # # corr_array[0,2] = GC_full_reduced_separate_regress(top_right, centre_ref, lag=5, alpha=.1)#(centre_ref, top_right)
#             # corr_array[0,0] = GC_full_reduced_separate_regress(centre_ref, top_left, lag=5, alpha=.1) #(centre_ref, top_left)
#             # corr_array[0,1] = GC_full_reduced_separate_regress(centre_ref, top_center, lag=5, alpha=.1)#(centre_ref, top_center)
#             # corr_array[0,2] = GC_full_reduced_separate_regress(centre_ref, top_right, lag=5, alpha=.1)#(centre_ref, top_right)
    
#             left = frame_b_[rr:rr+winsize,cc-winsize:cc].copy()
#             center = frame_b_[rr:rr+winsize,cc:cc+winsize].copy()
#             # center = frame_b_[rr-winsize//2:rr+winsize+winsize//2,cc-winsize//2:cc+winsize+winsize//2].copy()
#             # center = frame_b_[rr-winsize:rr+winsize+winsize,cc-winsize:cc+winsize+winsize].copy()
#             right = frame_b_[rr:rr+winsize,cc+winsize:cc+2*winsize].copy()
            
#             # # # corr_array[1,0] = spstats.pearsonr(centre_ref.ravel(), left.ravel())[0]
#             # # # corr_array[1,1] = spstats.pearsonr(centre_ref.ravel(), center.ravel())[0]
#             # # # corr_array[1,2] = spstats.pearsonr(centre_ref.ravel(), right.ravel())[0]
#             # # corr_array[1,0] = GC_full_reduced_separate_regress(left, centre_ref, lag=5, alpha=.1) #(centre_ref, left)
#             # # corr_array[1,1] = GC_full_reduced_separate_regress(center, centre_ref, lag=5, alpha=.1) #(centre_ref, center)
#             # # corr_array[1,2] = GC_full_reduced_separate_regress(right, centre_ref, lag=5, alpha=.1) #(centre_ref, right)
#             # corr_array[1,0] = GC_full_reduced_separate_regress(centre_ref, left, lag=5, alpha=.1) #(centre_ref, left)
#             # corr_array[1,1] = GC_full_reduced_separate_regress(centre_ref, center, lag=5, alpha=.1) #(centre_ref, center)
#             # corr_array[1,2] = GC_full_reduced_separate_regress(centre_ref, right, lag=5, alpha=.1) #(centre_ref, right)
            
#             bottom_left = frame_b_[rr+winsize:rr+2*winsize, cc-winsize:cc].copy()
#             bottom_center = frame_b_[rr+winsize:rr+2*winsize, cc:cc+winsize].copy()
#             bottom_right = frame_b_[rr+winsize:rr+2*winsize, cc+winsize:cc+2*winsize].copy()
            
#             # # # corr_array[2,0] = spstats.pearsonr(centre_ref.ravel(), bottom_left.ravel())[0]
#             # # # corr_array[2,1] = spstats.pearsonr(centre_ref.ravel(), bottom_center.ravel())[0]
#             # # # corr_array[2,2] = spstats.pearsonr(centre_ref.ravel(), bottom_right.ravel())[0]
#             # # corr_array[2,0] = GC_full_reduced_separate_regress(bottom_left, centre_ref, lag=5, alpha=.1) #(centre_ref, bottom_left)
#             # # corr_array[2,1] = GC_full_reduced_separate_regress(bottom_center, centre_ref, lag=5, alpha=.1)#(centre_ref, bottom_center)
#             # # corr_array[2,2] = GC_full_reduced_separate_regress(bottom_right, centre_ref, lag=5, alpha=.1)#(centre_ref, bottom_right)
#             # corr_array[2,0] = GC_full_reduced_separate_regress(centre_ref, bottom_left, lag=5, alpha=.1) #(centre_ref, bottom_left)
#             # corr_array[2,1] = GC_full_reduced_separate_regress(centre_ref, bottom_center, lag=5, alpha=.1)#(centre_ref, bottom_center)
#             # corr_array[2,2] = GC_full_reduced_separate_regress(centre_ref, bottom_right, lag=5, alpha=.1)#(centre_ref, bottom_right)
    
#             # corr_array[np.isnan(corr_array)] = 0 
#             # corr_array[1,1] = 0 #np.nanmean(corr_array[corr_array>0])
#             # corr_array = normxcorr2(centre_ref, center)
            
#             """
#             Generate the block correlation .... with sliding windows... 
#             """
            
#             # corr_array = GC_full_reduced_separate_regress_individual(centre_ref-centre_ref.mean(axis=-1)[...,None], 
#             #                                                          center-center.mean(axis=-1)[...,None], 
#             #                                                          lag=5, alpha=1) #(centre_ref, center)
#             corr_array = GC_full_reduced_separate_regress_individual(centre_ref, 
#                                                                      center, 
#                                                                      lag=5, alpha=1) #(centre_ref, center)
#             corr_array = corr_array.reshape((winsize,winsize))
#             # # # corr_array = granger_naive(centre_ref, center) # this scan needs to be completely reconfigured.... to accomodate for different sliding... 
#             # # # # corr_array = granger_naive2(centre_ref, center)
#             # # # # # corr_array[corr_array==0] = np.nan
#             # corr_array[corr_array>0] = 0
#             # corr_array = np.abs(corr_array)
            
#             # plt.figure()
#             # plt.imshow(corr_array)
#             # plt.show()
            
#             # # ### For PCCA maybe this below is best - check this reproduces... (if does we expand out of the block)
#             # # # corr_array_grad = np.array(np.gradient(np.abs(corr_array)))
#             # # corr_array_grad = np.array(np.gradient(corr_array)) # this finds homogeneity!!!,,, i see... 
#             # # mean_vector = np.nansum(corr_array_grad.reshape(2,-1), axis=-1) # this should be correct now. 
#             # # mean_vector = np.sum(corr_array) * mean_vector # add back the intensity... 
#             # YY, XX = np.indices(corr_array.shape)
#             # YY_ = YY - corr_array.shape[0]//2
#             # XX_ = XX - corr_array.shape[1]//2
            
#             # corr_array_vects = np.dstack([YY_,XX_])
#             # corr_array_vects = corr_array_vects/(np.linalg.norm(corr_array_vects, axis=-1)[...,None] + 1e-8)
            
#             # # # corr_vectors = corr_array_vects*np.clip(corr_array[...,None],0,1)
#             # corr_vectors = corr_array_vects*corr_array[...,None] 
#             # mean_vector = np.nanmean(corr_vectors.reshape(-1,2), axis=0) * np.sum(corr_array) 
            
#             mid = corr_array.shape[1]//2
            
#             corr_x_direction = -corr_array[:,:mid].sum() + corr_array[:,mid+1:].sum()
#             corr_y_direction = -corr_array[:mid].sum() + corr_array[mid+1:].sum()
#             intensity = np.sum(corr_array) #* np.sqrt(corr_x_direction**2 + corr_y_direction**2)
            
#             mean_vector = np.hstack([corr_y_direction, corr_x_direction])
#             mean_vector = mean_vector * intensity
#             # # # # # # simply the total strength of stuff flowing past the central pixel 
            
#             # # # """ this should still be correct for directionality ... """
#             # # mean_vector = np.hstack([np.nansum(corr_array[:,corr_array.shape[0]//2+1:]) - np.nansum(corr_array[:,:corr_array.shape[0]//2]),
#             # #                          np.nansum(corr_array[corr_array.shape[0]//2+1:]) - np.nansum(corr_array[:corr_array.shape[0]//2])])
#             # # mean_vector = np.sum(corr_array) * mean_vector
#             # # mean_vector = mean_vector[::-1]
#             # # # # out_vect[row_ii,row_jj,:] = mean_vector[::-1]
#             out_vect[row_ii,row_jj,:] = -mean_vector
            
#             row_jj += 1
#         row_ii +=1
            
#     import scipy.ndimage as ndimage
    
#     # # now we  apply smoothing to derive the flows!. ---> this is quite important!. 
#     out_vect[...,0] = ndimage.gaussian_filter(out_vect[...,0], sigma=1)
#     out_vect[...,1] = ndimage.gaussian_filter(out_vect[...,1], sigma=1)
    
#     # plot the vector field
#     sampling = 1
    
#     plt.figure(figsize=(15,15))
#     plt.imshow(myVid[1])
#     plt.quiver(xy_coords[::sampling,::sampling,0], 
#                 xy_coords[::sampling,::sampling,1], 
#                 out_vect[::sampling,::sampling,1],  # x 
#                 -out_vect[::sampling,::sampling,0]) # y 
#     plt.show()
    

#     # plt.figure(figsize=(15,15))
#     # # plt.imshow(myVid[1])
#     # plt.quiver(XX, 
#     #            YY, 
#     #             corr_array_vects[...,1], 
#     #             corr_array_vects[...,0])
#     # plt.show()


# # # #myVidGrey = np.zeros([myVid.shape[0], myVid.shape[1], myVid.shape[2]])
# # # myVidGrey = np.zeros([194, 60, 90 ])
# # # for i in range(0, myVid.shape[0]):
# # #     myGrey = color.rgb2gray(myVid[i,:,:,])
# # #     myGreyRescale = rescale(myGrey, .125)
# # #     myVidGrey[i,:,:] = myGreyRescale


# """
# This is correct!..... 
# """
# optical_flow_params = dict(pyr_scale=0.5, levels=1, winsize=5, iterations=5, poly_n=3, poly_sigma=1.2, flags=0)
 
# vid_flow = extract_optflow(255*myVid[start:end][:], 
#                             optical_flow_params, 
#                             rescale_intensity=False, 
#                             intensity_range=[2,98])

# YY, XX = np.indices(myVid.shape[1:3])
# sampling = 3 

# plt.figure(figsize=(15,15))
# plt.subplot(121)
# plt.imshow(myVid[1])
# # plt.quiver(xy_coords[:,:,0], 
# #             xy_coords[:,:,1], 
# #             out_vect[:,:,1], 
# #             -out_vect[:,:,0])
# plt.quiver(xy_coords[:,:,0], 
#             xy_coords[:,:,1], 
#             out_vect[:,:,1], 
#             -out_vect[:,:,0])
# plt.subplot(122)
# plt.imshow(myVid[1,:,:])
# # plt.quiver(XX[::sampling,::sampling], 
# #             YY[::sampling,::sampling], 
# #             np.nanmean(vid_flow[:20], axis=0)[::sampling,::sampling,0], 
# #             -np.nanmean(vid_flow[:20],axis=0)[::sampling,::sampling,1])
# plt.quiver(XX[::sampling,::sampling], 
#             YY[::sampling,::sampling], 
#             vid_flow.mean(axis=0)[::sampling,::sampling,0], 
#             -vid_flow.mean(axis=0)[::sampling,::sampling,1])
# plt.show()






    
# plt.imshow(myVidGrey[1,:,:])
# plt.title('Example Snapshot of GreScale Video')

# myMat = myVidGrey
# transferEntropyVector = np.zeros([myMat.shape[1], myMat.shape[2], 2])

# myK = 1
# thresh = 20

# from tqdm import tqdm 

# ##for each  valid box in matCellRatioT compute transfer entropy with lag 1
# for i in tqdm(range(1,myMat.shape[1] -1)):
#     # print(i)
#     for j in range(1,myMat.shape[2] -1):
#         # print(j)
#         if (sum(myMat[:,i,j] > 0) > thresh):
#             xs = myMat[1:20, i,j]
#             xs = (xs - xs.mean()) / xs.std()
            
#             if  (sum(myMat[:, i+1,j] > 0) > thresh):
#                 ys = myMat[2:21, (i+1), j]
#                 ys = (ys - ys.mean()) / ys.std()
#                 #newVal = np.array([1,0]) * transfer_entropy(xs, ys, k= 1)
#                 newVal = np.array([1,0]) * te.te_compute(xs, ys,  1, 5)
#                 transferEntropyVector[i,j,:] = transferEntropyVector[i,j,:] + newVal
#             if  (sum(myMat[:, i-1,j] > 0) > thresh):
#                 ys = myMat[2:21, (i-1), j]
#                 ys = (ys - ys.mean()) / ys.std()
#                 #newVal = np.array([-1,0]) * transfer_entropy(xs, ys, k = 1)
#                 newVal = np.array([-1,0]) * te.te_compute(xs, ys, 1, 5)
#                 transferEntropyVector[i,j,:] = transferEntropyVector[i,j,:] + newVal
#             if  (sum(myMat[:, i,j+1] > 0) > thresh):
#                 ys = myMat[2:21, i, (j+1)]
#                 ys = (ys - ys.mean()) / ys.std()
#                 #newVal = np.array([0,1]) * transfer_entropy(xs, ys, k= 1)
#                 newVal = np.array([0,1]) * te.te_compute(xs, ys, 1, 5)
#                 transferEntropyVector[i,j,:] = transferEntropyVector[i,j,:] + newVal   
#             if  (sum(myMat[:, i,j-1] > 0) > thresh):
#                 ys = myMat[2:21, i, (j-1)]
#                 ys = (ys - ys.mean()) / ys.std()
#                 #newVal = np.array([0,-1]) * transfer_entropy(xs, ys, k=1)
#                 newVal = np.array([0,-1]) * te.te_compute(xs, ys, 1, 5)
#                 transferEntropyVector[i,j,:] = transferEntropyVector[i,j,:] + newVal   

# plt.imshow(myMat[20,:,:])
# plt.title('Rac: Frame from Ratiometric Movie')


# YY, XX = np.indices(myVidGrey.shape[1:3])

# plt.figure()
# plt.imshow(myMat[1,:,:])
# plt.quiver(XX, 
#            YY, 
#            transferEntropyVector[:,:,0], 
#            -transferEntropyVector[:,:,1], scale=50)
# plt.show()

# ######
# transferEntropyVector[np.isnan(transferEntropyVector)] = 0               
# # compute the cumulative gradients of transfer entropy gradient
# gradx = np.gradient(transferEntropyVector[:,:,0])
# gradxSum = np.array(abs(gradx[0]) + abs(gradx[1]))
# grady = np.gradient(transferEntropyVector[:,:,1])
# gradySum = np.array(abs(grady[0]) + abs(grady[1]))
# totalGradient = (gradxSum + gradySum)
# plt.imshow(totalGradient)
# plt.title('Sum of abs (xGradient), abs(yGradient)')
# #clean up weird grad
# totalGradient[totalGradient >50] = 0

# #totalGradient[totalGradient < 10] = 0
# #Use difference of Gaussian and treshholding to determine microdomains
# differenceOfGaussian = gaussian_filter(totalGradient, 1) - gaussian_filter(totalGradient, 20)
# differenceOfGaussian[differenceOfGaussian < (np.mean(differenceOfGaussian) + 3 * np.std(differenceOfGaussian))] = 0
# plt.imshow(differenceOfGaussian, interpolation = 'nearest')
# plt.title('Rac')

# plt.show()

# #plt.imshow(differenceOfGaussian, interpolation = 'nearest')
# #plt.show()
# #########################################################
# plt.imshow(abs(transferEntropyVector[:,:,1]) + abs(transferEntropyVector[:,:,0]))
# plt.title('Sum of absolute value of TE vector in both X and Y direction, Rac Cell1')


