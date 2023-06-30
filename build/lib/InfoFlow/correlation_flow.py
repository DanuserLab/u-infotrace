

def normxcorr2(A,B, norm=True, mode='same'):

    from scipy.signal import correlate2d
    from skimage.feature import match_template
        
    A_ = A-np.nanmean(A)
    B_ = B-np.nanmean(B)
        
    return correlate2d(A_,B_, mode=mode)

# def nd_xcorr_lag(img1, img2, lag=1, mode='same'):
# def nd_xcorr_lag(img1, img2, lag=1, mode='same'):
# 	from scipy.signal import correlate

#  	img1_ = img1[...,lag:img1.shape[-1]].copy()
#  	img2_ = img2[...,:img2.shape[-1]-lag].copy()
#     # img1_ = img1[...,img1.shape[-1]-lag].copy()
# # 	img2_ = img2[...,lag:img2.shape[-1]].copy()

#  	corr_a_b = correlate(img1_-img1_.mean(axis=-1)[...,None], 
# 	                     img2_-img2_.mean(axis=-1)[...,None], mode=mode)
# 	corr_a_b = corr_a_b.mean(axis=-1) # integrate over time!. 

# 	return corr_a_b
        
def nd_xcorr_lag(img1, img2, lag=1, mode='same', demean=False):
        
    from scipy.signal import correlate
    import numpy as np 

    img1_ = img1[...,lag:img1.shape[-1]].copy()
    img2_ = img2[...,:img2.shape[-1]-lag].copy()

#     corr_a_b = correlate(img1_-img1_.mean(axis=-1)[...,None], 
#                          img2_-img2_.mean(axis=-1)[...,None], mode=mode)
    # corr_a_b = correlate(img1_, img2_, mode=mode)
    if demean:
        corr_a_b = correlate(img2_-img2_.mean(axis=-1)[...,None], 
                             img1_-img1_.mean(axis=-1)[...,None], 
                             mode=mode)
    else:
        corr_a_b = correlate(img2_, 
                             img1_, mode=mode)
    # corr_a_b = corr_a_b.mean(axis=-1) # integrate over time!. 
    corr_a_b = np.nanmax(corr_a_b, axis=-1)
    
    return corr_a_b