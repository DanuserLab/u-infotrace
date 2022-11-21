#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:22:15 2022

@author: s434626
"""

    
def get_field_shape(image_size, search_area_size, overlap):
    """Compute the shape of the resulting flow field.
    Given the image size, the interrogation window size and
    the overlap size, it is possible to calculate the number
    of rows and columns of the resulting flow field.
    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns, easy to obtain using .shape
    search_area_size: tuple
        the size of the interrogation windows (if equal in frames A,B)
        or the search area (in frame B), the largest  of the two
    overlap: tuple
        the number of pixel by which two adjacent interrogation
        windows overlap.
    Returns
    -------
    field_shape : 2-element tuple
        the shape of the resulting flow field
    """
    field_shape = (np.array(image_size) - np.array(search_area_size)) // (
        np.array(search_area_size) - np.array(overlap)
    ) + 1
    
    return field_shape


def get_coordinates(image_size, search_area_size, overlap, center_on_field = True):
    """Compute the x, y coordinates of the centers of the interrogation windows.
    the origin (0,0) is like in the image, top left corner
    positive x is an increasing column index from left to right
    positive y is increasing row index, from top to bottom
    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.
    search_area_size: int
        the size of the search area windows, sometimes it's equal to
        the interrogation window size in both frames A and B
    overlap: int = 0 (default is no overlap)
        the number of pixel by which two adjacent interrogation
        windows overlap.
    Returns
    -------
    x : 2d np.ndarray
        a two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.
    y : 2d np.ndarray
        a two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.
        Coordinate system 0,0 is at the top left corner, positive
        x to the right, positive y from top downwards, i.e.
        image coordinate system
    """

    # get shape of the resulting flow field
    field_shape = get_field_shape(image_size,
                                  search_area_size,
                                  overlap)
    # print(len(field_shape))
    # print(field_shape)
    # compute grid coordinates of the search area window centers
    # note the field_shape[1] (columns) for x
    x = (
        np.arange(field_shape[1]) * (search_area_size - overlap)
        + (search_area_size) / 2.0
    )
    # note the rows in field_shape[0]
    y = (
        np.arange(field_shape[0]) * (search_area_size - overlap)
        + (search_area_size) / 2.0
    )

    # moving coordinates further to the center, so that the points at the
    # extreme left/right or top/bottom
    # have the same distance to the window edges. For simplicity only integer
    # movements are allowed.
    if center_on_field == True:
        x += (
            image_size[1]
            - 1
            - ((field_shape[1] - 1) * (search_area_size - overlap) +
                (search_area_size - 1))
        ) // 2
        y += (
            image_size[0] - 1
            - ((field_shape[0] - 1) * (search_area_size - overlap) +
               (search_area_size - 1))
        ) // 2

        # the origin 0,0 is at top left
        # the units are pixels

    return np.meshgrid(x, y)

def get_rect_coordinates(frame_a, window_size, overlap, center_on_field = False):
    '''
    Rectangular grid version of get_coordinates.
    '''
    if isinstance(window_size, tuple) == False and isinstance(window_size, list) == False:
        window_size = [window_size, window_size]
    if isinstance(overlap, tuple) == False and isinstance(overlap, list) == False:
        overlap = [overlap, overlap]
        
    _, y = get_coordinates(frame_a, window_size[0], overlap[0], center_on_field = False)
    x, _ = get_coordinates(frame_a, window_size[1], overlap[1], center_on_field = False)
    
    return np.meshgrid(x[0,:], y[:,0])



def sliding_window_array(image, window_size = 64, overlap = 32):
    '''
    This version does not use numpy as_strided and is much more memory efficient.
    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in
    which each slice, (along the first axis) is an interrogation window. 
    '''
    import numpy as np 
    
    if isinstance(window_size, tuple) == False and isinstance(window_size, list) == False:
        window_size = [window_size, window_size]
    if isinstance(overlap, tuple) == False and isinstance(overlap, list) == False:
        overlap = [overlap, overlap]

    x, y = get_rect_coordinates(image.shape, window_size, overlap, center_on_field = False)
    x = (x - window_size[1]//2).astype(int); y = (y - window_size[0]//2).astype(int)
    x, y = np.reshape(x, (-1,1,1)), np.reshape(y, (-1,1,1))

    win_x, win_y = np.meshgrid(np.arange(0, window_size[1]), np.arange(0, window_size[0]))
    win_x = win_x[np.newaxis,:,:] + x
    win_y = win_y[np.newaxis,:,:] + y
    
    # print(win_x.shape, win_y.shape)
    windows = image[win_y, win_x]
    
    return windows


def sliding_window_array_time(image, window_size = 64, overlap = 32):
    '''
    This version does not use numpy as_strided and is much more memory efficient.
    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in
    which each slice, (along the first axis) is an interrogation window. 
    '''
    import numpy as np 
    
    if isinstance(window_size, tuple) == False and isinstance(window_size, list) == False:
        window_size = [window_size, window_size]
    if isinstance(overlap, tuple) == False and isinstance(overlap, list) == False:
        overlap = [overlap, overlap]

    x, y = get_rect_coordinates(image.shape[:-1], window_size, overlap, center_on_field = False)
    x = (x - window_size[1]//2).astype(int); y = (y - window_size[0]//2).astype(int)
    x, y = np.reshape(x, (-1,1,1)), np.reshape(y, (-1,1,1))

    win_x, win_y = np.meshgrid(np.arange(0, window_size[1]), np.arange(0, window_size[0]))
    win_x = win_x[np.newaxis,:,:] + x
    win_y = win_y[np.newaxis,:,:] + y
    
    print(win_x.shape, win_y.shape)
    windows = image[win_y, win_x]
    
    return windows


def moving_window_array(array, window_size, overlap):
    """
    This is a nice numpy trick. The concept of numpy strides should be
    clear to understand this code.
    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in
    which each slice, (along the first axis) is an interrogation window.
    """
    import numpy as np 
    
    sz = array.itemsize
    shape = array.shape
    array = np.ascontiguousarray(array)

    strides = (
        sz * shape[1] * (window_size - overlap),
        sz * (window_size - overlap),
        sz * shape[1],
        sz,
    )
    shape = (
        int((shape[0] - window_size) / (window_size - overlap)) + 1,
        int((shape[1] - window_size) / (window_size - overlap)) + 1,
        window_size,
        window_size,
    )

    return np.lib.stride_tricks.as_strided(
        array, strides=strides, shape=shape
    ).reshape(-1, window_size, window_size)


def moving_window_array_time(array, window_size, overlap):
    """
    This is a nice numpy trick. The concept of numpy strides should be
    clear to understand this code.
    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in
    which each slice, (along the first axis) is an interrogation window.
    """
    import numpy as np 
    
    sz = array.itemsize
    shape = array.shape
    array = np.ascontiguousarray(array)

    strides = (
        sz * shape[1] * (window_size - overlap),
        sz * (window_size - overlap),
        sz * shape[1],
        sz,
    )
    shape = (
        int((shape[0] - window_size) / (window_size - overlap)) + 1,
        int((shape[1] - window_size) / (window_size - overlap)) + 1,
        window_size,
        window_size,
    )

    return np.lib.stride_tricks.as_strided(
        array, strides=strides, shape=shape
    ).reshape(-1, window_size, window_size)


def read_video_cv2(avifile):
    
    import cv2
    
    vidcap = cv2.VideoCapture(avifile)
    success,image = vidcap.read()
    
    vid_array = []
    
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        if success:
            vid_array.append(image)
        count += 1
        
    vid_array = np.array(vid_array)
      
    return vid_array


def causal_flow(vid, cause_fnc, winsize=3, **kwargs):

    import scipy.ndimage as ndimage 
    from tqdm import tqdm 
    
    frame_a_ = np.pad(vid.transpose(1,2,0), [[winsize,winsize], [winsize,winsize], [0,0]], mode='constant', constant_values=0)
    
    # windows = moving_window_array(frame_a[:,:,0], window_size=winsize, overlap=winsize//2)
    # windows_2 = sliding_window_array(frame_a[:,:,0], window_size=winsize, overlap=winsize//2)
    windows_3 = sliding_window_array_time(frame_a_, window_size=(3*winsize), overlap=2*winsize)
    # windows_3 = sliding_window_array_time(frame_a_, window_size=(3*winsize), overlap=winsize//2)
    x, y = get_coordinates(image_size=frame_a_[:,:,0].shape, 
                            search_area_size=(3*winsize), overlap=2*winsize) # so this is correct !. 
    # x, y = get_coordinates(image_size=frame_a_[:,:,0].shape, 
    #                        search_area_size=(3*winsize), overlap=winsize//2) # so this is correct !. 
    
    # windows_3 = sliding_window_array_time(frame_a_, window_size=(1*winsize), overlap=0)
    # # windows_3 = sliding_window_array_time(frame_a_, window_size=(3*winsize), overlap=winsize//2)
    # x, y = get_coordinates(image_size=frame_a_[:,:,0].shape, 
    #                         search_area_size=(1*winsize), overlap=0) # so this is correct !. 
    xy_coords = np.dstack([x,y])
    
    GC_vectors = []
    
    # for ii in tqdm(np.arange(len(windows_3))):
    for ii in tqdm(np.arange(len(windows_3))):
        
        """
        This should allow usage of all functions that have the same call signature.
        """
        corr_array = cause_fnc(windows_3[ii, winsize:2*winsize, winsize:2*winsize], 
                               windows_3[ii, winsize:2*winsize, winsize:2*winsize], **kwargs) 
        # corr_array[1,1] = 0
        # corr_array = GC_full_reduced_separate_regress_individual(windows_3[ii], 
        #                                                           windows_3[ii], 
        #                                                               lag=lag, alpha=alpha) #(centre_ref, center)
        corr_array = corr_array.reshape((winsize,winsize))
        mid = corr_array.shape[1]//2
            
        corr_x_direction = -np.nansum(corr_array[:,:mid]) + np.nansum(corr_array[:,mid+1:])
        corr_y_direction = -np.nansum(corr_array[:mid]) + np.nansum(corr_array[mid+1:])
        intensity = np.nansum(corr_array) #* np.sqrt(corr_x_direction**2 + corr_y_direction**2)
        
        mean_vector = np.hstack([corr_y_direction, corr_x_direction])
        mean_vector = mean_vector * intensity
        
        GC_vectors.append(-mean_vector)
            
    GC_vectors = np.array(GC_vectors).reshape((xy_coords.shape))
    GC_vectors[...,0] = ndimage.gaussian_filter(GC_vectors[...,0], sigma=1.)
    GC_vectors[...,1] = ndimage.gaussian_filter(GC_vectors[...,1], sigma=1.)
    
    GC_vectors = np.dstack([sktform.resize(GC_vectors[...,ch], output_shape=vid.shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
    
    return GC_vectors


def causal_flow_scores(vid, cause_fnc, winsize=3, **kwargs):

    import scipy.ndimage as ndimage 
    from tqdm import tqdm 
    
    frame_a_ = np.pad(vid.transpose(1,2,0), [[winsize,winsize], [winsize,winsize], [0,0]], mode='constant', constant_values=0)
    
    # windows = moving_window_array(frame_a[:,:,0], window_size=winsize, overlap=winsize//2)
    # windows_2 = sliding_window_array(frame_a[:,:,0], window_size=winsize, overlap=winsize//2)
    windows_3 = sliding_window_array_time(frame_a_, window_size=(3*winsize), overlap=2*winsize)
    # windows_3 = sliding_window_array_time(frame_a_, window_size=(3*winsize), overlap=winsize//2)
    x, y = get_coordinates(image_size=frame_a_[:,:,0].shape, 
                            search_area_size=(3*winsize), overlap=2*winsize) # so this is correct !. 
    # x, y = get_coordinates(image_size=frame_a_[:,:,0].shape, 
    #                        search_area_size=(3*winsize), overlap=winsize//2) # so this is correct !. 
    
    # windows_3 = sliding_window_array_time(frame_a_, window_size=(1*winsize), overlap=0)
    # # windows_3 = sliding_window_array_time(frame_a_, window_size=(3*winsize), overlap=winsize//2)
    # x, y = get_coordinates(image_size=frame_a_[:,:,0].shape, 
    #                         search_area_size=(1*winsize), overlap=0) # so this is correct !. 
    xy_coords = np.dstack([x,y])
    
    GC_scores = []
    
    # for ii in tqdm(np.arange(len(windows_3))):
    for ii in tqdm(np.arange(len(windows_3))):
        
        """
        This should allow usage of all functions that have the same call signature.
        """
        corr_array = cause_fnc(windows_3[ii, winsize:2*winsize, winsize:2*winsize], 
                               windows_3[ii, winsize:2*winsize, winsize:2*winsize], **kwargs) 
        # corr_array[1,1] = 0
        # corr_array = GC_full_reduced_separate_regress_individual(windows_3[ii], 
        #                                                           windows_3[ii], 
        #                                                               lag=lag, alpha=alpha) #(centre_ref, center)
        corr_array = corr_array.reshape((winsize,winsize))
        
        GC_scores.append(corr_array)
        # mid = corr_array.shape[1]//2
            
        # corr_x_direction = -np.nansum(corr_array[:,:mid]) + np.nansum(corr_array[:,mid+1:])
        # corr_y_direction = -np.nansum(corr_array[:mid]) + np.nansum(corr_array[mid+1:])
        # intensity = np.nansum(corr_array) #* np.sqrt(corr_x_direction**2 + corr_y_direction**2)
        
        # mean_vector = np.hstack([corr_y_direction, corr_x_direction])
        # mean_vector = mean_vector * intensity
        
        # GC_vectors.append(-mean_vector)
            
    GC_scores = np.array(GC_scores).reshape((xy_coords.shape[:-1])+(winsize, winsize))
    # GC_vectors[...,0] = ndimage.gaussian_filter(GC_vectors[...,0], sigma=1.)
    # GC_vectors[...,1] = ndimage.gaussian_filter(GC_vectors[...,1], sigma=1.)
    
    # GC_vectors = np.dstack([sktform.resize(GC_vectors[...,ch], output_shape=vid.shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
    GC_scores = sktform.resize(GC_scores,  output_shape=vid.shape[1:]+(winsize,winsize), preserve_range=True, order=1)
    
    return GC_scores


def causal_block_flow_scores_gradient(scores):
    
    """
    scores : M x N x winsize x winsize 
    """
    import numpy as np 
    # collapse the combine all into one into vectors.
    mid = scores.shape[-1]//2
            
    corr_x_direction = -np.apply_over_axes(np.nansum, scores[:,:,:,:mid], [-1,-2]) + np.apply_over_axes(np.nansum, scores[:,:,:,mid+1:], [-1,-2])
    corr_y_direction = -np.apply_over_axes(np.nansum, scores[:,:,:mid], [-1,-2]) + np.apply_over_axes(np.nansum, scores[:,:,mid+1:], [-1,-2])
    intensity = np.nansum(scores.reshape(scores.shape[0], scores.shape[1], -1), axis=-1) 
    
    mean_vector = np.array([np.squeeze(corr_y_direction), 
                            np.squeeze(corr_x_direction)])
    mean_vector = mean_vector * intensity[None,...]
    mean_vector = mean_vector.transpose(1,2,0)
    mean_vector = -mean_vector

    return mean_vector

"""
For use with PCCA!.
"""
def causal_block_flow(vid, cause_fnc, winsize=3, **kwargs):

    import scipy.ndimage as ndimage 
    from tqdm import tqdm 
    
    frame_a_ = np.pad(vid.transpose(1,2,0), [[winsize,winsize], [winsize,winsize], [0,0]], mode='constant', constant_values=0)
    
    # windows = moving_window_array(frame_a[:,:,0], window_size=winsize, overlap=winsize//2)
    # windows_2 = sliding_window_array(frame_a[:,:,0], window_size=winsize, overlap=winsize//2)
    windows_3 = sliding_window_array_time(frame_a_, window_size=(3*winsize), overlap=2*winsize)
    # windows_3 = sliding_window_array_time(frame_a_, window_size=(3*winsize), overlap=winsize//2)
    x, y = get_coordinates(image_size=frame_a_[:,:,0].shape, 
                            search_area_size=(3*winsize), overlap=2*winsize) # so this is correct !. 
    # x, y = get_coordinates(image_size=frame_a_[:,:,0].shape, 
    #                        search_area_size=(3*winsize), overlap=winsize//2) # so this is correct !. 
    
    # windows_3 = sliding_window_array_time(frame_a_, window_size=(1*winsize), overlap=0)
    # # windows_3 = sliding_window_array_time(frame_a_, window_size=(3*winsize), overlap=winsize//2)
    # x, y = get_coordinates(image_size=frame_a_[:,:,0].shape, 
    #                         search_area_size=(1*winsize), overlap=0) # so this is correct !. 
    xy_coords = np.dstack([x,y])
    
    GC_vectors = []
    
    # for ii in tqdm(np.arange(len(windows_3))):
    for ii in tqdm(np.arange(len(windows_3))):
        
        """
        This should allow usage of all functions that have the same call signature.
        """
        corr_array = cause_fnc(windows_3[ii, :, :], 
                               windows_3[ii, :, :], **kwargs) 
        # corr_array = GC_full_reduced_separate_regress_individual(windows_3[ii], 
        #                                                           windows_3[ii], 
        #                                                               lag=lag, alpha=alpha) #(centre_ref, center)
        corr_array = corr_array.reshape((winsize,winsize))
        mid = corr_array.shape[1]//2
            
        corr_x_direction = -np.nansum(corr_array[:,:mid]) + np.nansum(corr_array[:,mid+1:])
        corr_y_direction = -np.nansum(corr_array[:mid]) + np.nansum(corr_array[mid+1:])
        intensity = np.sum(corr_array) #* np.sqrt(corr_x_direction**2 + corr_y_direction**2)
        
        mean_vector = np.hstack([corr_y_direction, corr_x_direction])
        mean_vector = mean_vector * intensity
        
        GC_vectors.append(mean_vector) # when is this negative? 
            
        
    GC_vectors = np.array(GC_vectors).reshape((xy_coords.shape))
    GC_vectors[...,0] = ndimage.gaussian_filter(GC_vectors[...,0], sigma=1.)
    GC_vectors[...,1] = ndimage.gaussian_filter(GC_vectors[...,1], sigma=1.)
    
    GC_vectors = np.dstack([sktform.resize(GC_vectors[...,ch], output_shape=vid.shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
    
    
    return GC_vectors


def gaussian_video_pyramid(vid, scales=[1,2,4,8], sigma=1):
    
    import skimage.transform as sktform
    import numpy as np
    from scipy.ndimage import gaussian_filter
    
    
    # normalise the video if not. 
    vid_ = (vid - vid.min())/(vid.max()-vid.min())
    
    if sigma>0:
        vids = [ndimage.gaussian_filter(sktform.resize(vid_, output_shape=(vid_.shape[0], vid_.shape[1]//s, vid_.shape[2]//s), preserve_range=True), sigma=sigma) for s in scales]
    else:
        # no smoothing
        vids = [sktform.resize(vid_, output_shape=(vid_.shape[0], vid_.shape[1]//s, vid_.shape[2]//s), preserve_range=True) for s in scales]
   
    return vids


def laplacian_video_pyramid(vid, scales=[2,4,8], sigma=1):
    
    import skimage.transform as sktform
    import numpy as np
    from scipy.ndimage import gaussian_filter
    
    
    # normalise the video if not. 
    vid_ = (vid - vid.min())/(vid.max()-vid.min())
    
    vids_laplace = []
    vids_blur = [vid_]
        
    for ii in np.arange(len(scales)):   
        s = scales[ii]
        ds_im = sktform.resize(vid_, output_shape=(vid_.shape[0], vid_.shape[1]//s, vid_.shape[2]//s), preserve_range=True)
        ds_im = ndimage.gaussian_filter(ds_im, sigma=1) 
        
        # if ii == len(scales)-1:
        #     vids_laplace.append(ds_im)
        # else:
        diff = vids_blur[-1] - sktform.resize(ds_im, output_shape=vids_blur[-1].shape, preserve_range=True)
        vids_laplace.append(diff)
        
        vids_blur.append(ds_im)
    vids_laplace.append(ds_im)
        
    return vids_laplace

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

    """
    Imports of various flow functions. 
    """
    # from InfoFlow.gc_flow import GC_full_reduced_separate_regress_individual
    # from InfoFlow.DDC_flow import DDC_cause
    # from InfoFlow.pdc_dtf_flow import PDC_central_flow
    # from InfoFlow.pcca_flow import pcca_cause_block
    # from InfoFlow.correlation_flow import nd_xcorr_lag
    # from InfoFlow.LK_flow import Linear_LK_cause
    from gc_flow import GC_full_reduced_separate_regress_individual
    from DDC_flow import DDC_cause
    from pdc_dtf_flow import PDC_central_flow
    from pcca_flow import pcca_cause_block
    from correlation_flow import nd_xcorr_lag
    from optical_flow import extract_optflow
    from LK_flow import Linear_LK_cause
    from flow_vis import flow_to_color
    
    from dynamic_image import _compute_dynamic_image
    
    
    """
    Write a script to get the dynamic image. 
    """
    

    #def rgb2gray(rgb):
    
    #   r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    #    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    #    return gray
    # myVid = read_video_cv2(r'../821-10_l.mov') #works ! # weird ? 
    # myVid = read_video_cv2(r'../9-19_l.mov') #works !
    # myVid =  read_video_cv2(r'../001-0436.avi') # works!
    myVid = read_video_cv2(r'../3687-18_70.mov') #### spurious arrows present for the LK flow. --- is this due to the implementation of the estimation? # can we instead use properly the maximum likelihood estimate? 
    # myVid = read_video_cv2(r'../620-72_l.mov')
    # myVid = read_video_cv2(r'../637-147_l.mov')
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
    
    dyn_image_Vid = _compute_dynamic_image(myVid[...,None])
    
    plt.figure(figsize=(5,5))
    plt.title('Dynamic Image')
    plt.imshow(dyn_image_Vid, cmap='gray')
    plt.show()

    plt.figure(figsize=(5,5))
    plt.title('Mean Image')
    plt.imshow(np.nanmean(myVid, axis=0), cmap='gray')
    plt.show()
    
    plt.figure(figsize=(5,5))
    plt.title('Max Image')
    plt.imshow(np.nanmax(myVid, axis=0), cmap='gray')
    plt.show()
    
    
    """
    Are there other ways to downsample? - do we need to add smoothing after each bilinear interpolation downsample? 
    """
    # myVid = sktform.resize(myVid, output_shape=(myVid.shape[0], myVid.shape[1], myVid.shape[2]), preserve_range=True)
    # myVid2 = sktform.resize(myVid, output_shape=(myVid.shape[0], myVid.shape[1]//2, myVid.shape[2]//2), preserve_range=True)
    # myVid4 = sktform.resize(myVid, output_shape=(myVid.shape[0], myVid.shape[1]//4, myVid.shape[2]//4), preserve_range=True)
    # myVid8 = sktform.resize(myVid, output_shape=(myVid.shape[0], myVid.shape[1]//8, myVid.shape[2]//8), preserve_range=True)
    
    myVid, myVid2, myVid4, myVid8 = gaussian_video_pyramid(myVid, scales=[1,2,4,8], sigma=1)
    # myVid, myVid2, myVid4, myVid8 = laplacian_video_pyramid(myVid, scales=[2,4,8], sigma=1)
    
    
# =============================================================================
#     1. extract all windowed - flat
# =============================================================================
    
    # # this is normal GC. 
    # GC_vectors = causal_flow(myVid, GC_full_reduced_separate_regress_individual, winsize=3, lag=5, alpha=1)
    # GC_vectors2 = causal_flow(myVid2, GC_full_reduced_separate_regress_individual, winsize=3, lag=5, alpha=1)
    # GC_vectors4 = causal_flow(myVid4, GC_full_reduced_separate_regress_individual, winsize=3, lag=5, alpha=1)
    # GC_vectors8 = causal_flow(myVid8, GC_full_reduced_separate_regress_individual, winsize=3, lag=5, alpha=1)
    
    GC_vectors = causal_flow_scores(myVid, GC_full_reduced_separate_regress_individual, winsize=3, lag=5, alpha=1)
    GC_vectors2 = causal_flow_scores(myVid2, GC_full_reduced_separate_regress_individual, winsize=3, lag=5, alpha=1)
    GC_vectors4 = causal_flow_scores(myVid4, GC_full_reduced_separate_regress_individual, winsize=3, lag=5, alpha=1)
    GC_vectors8 = causal_flow_scores(myVid8, GC_full_reduced_separate_regress_individual, winsize=3, lag=5, alpha=1)
    
    # # this is DDC
    # GC_vectors = causal_flow(myVid, DDC_cause, eps=1e-12, alpha=1e-2)
    # GC_vectors2 = causal_flow(myVid2, DDC_cause, eps=1e-12, alpha=1e-2)
    # GC_vectors4 = causal_flow(myVid4, DDC_cause, eps=1e-12, alpha=1e-2)
    # GC_vectors8 = causal_flow(myVid8, DDC_cause, eps=1e-12, alpha=1e-2)
    
    # GC_vectors = causal_flow_scores(myVid, DDC_cause, eps=1e-12, alpha=1e-2)
    # GC_vectors2 = causal_flow_scores(myVid2, DDC_cause, eps=1e-12, alpha=1e-2)
    # GC_vectors4 = causal_flow_scores(myVid4, DDC_cause, eps=1e-12, alpha=1e-2)
    # GC_vectors8 = causal_flow_scores(myVid8, DDC_cause, eps=1e-12, alpha=1e-2)
    
    # # # linear LK 
    # # # # GC = Linear_LK_cause
    # GC_vectors = causal_flow(myVid, Linear_LK_cause, eps=1e-12)
    # GC_vectors2 = causal_flow(myVid2, Linear_LK_cause, eps=1e-12)
    # GC_vectors4 = causal_flow(myVid4, Linear_LK_cause, eps=1e-12)
    # GC_vectors8 = causal_flow(myVid8, Linear_LK_cause, eps=1e-12)
    
    # GC_vectors = causal_flow_scores(myVid, Linear_LK_cause, eps=1e-12)
    # GC_vectors2 = causal_flow_scores(myVid2, Linear_LK_cause, eps=1e-12)
    # GC_vectors4 = causal_flow_scores(myVid4, Linear_LK_cause, eps=1e-12)
    # GC_vectors8 = causal_flow_scores(myVid8, Linear_LK_cause, eps=1e-12)
    
#     # this is the PDC --- very very slow!. 
#     # print('computing original resolution ...')
#     # GC_vectors = causal_flow(myVid, PDC_central_flow, lag=5, alpha=1e-2) # this seems very slow? 2hr!!!! 
#     # print('computing 2x downsample resolution ...')
#     # GC_vectors2 = causal_flow(myVid2, PDC_central_flow, lag=5, alpha=1e-2)
#     # print('computing 4x downsample resolution ...')
#     # GC_vectors4 = causal_flow(myVid4, PDC_central_flow, lag=5, alpha=1e-2)
#     # print('computing 8x downsample resolution ...')
#     # GC_vectors8 = causal_flow(myVid8, PDC_central_flow, lag=5, alpha=1e-2)
    
    
#     # # # this is the correlation flow
#     # GC_vectors = causal_flow(myVid, nd_xcorr_lag, lag=1, mode='same')
#     # GC_vectors2 = causal_flow(myVid2, nd_xcorr_lag, lag=1, mode='same')
#     # GC_vectors4 = causal_flow(myVid4, nd_xcorr_lag, lag=1, mode='same')
#     # GC_vectors8 = causal_flow(myVid8, nd_xcorr_lag, lag=1, mode='same')
    
    
#     # now we need to do e.g. 
    
    
#     # # # PCCA flow. 
#     # # GC_vectors = causal_block_flow(myVid, pcca_cause_block, 
#     # #                                block_size=3,
#     # #                                k=1, 
#     # #                                m=3, 
#     # #                                eta_xt=5e-4, 
#     # #                                eta_yt=5e-4,
#     # #                                eta_xtkm=5e-4) # this seems very slow? 2hr!!!! 
    
#     # GC_vectors2 = causal_block_flow(myVid2, pcca_cause_block, 
#     #                                block_size=3,
#     #                                k=1, 
#     #                                m=3, 
#     #                                eta_xt=5e-4, 
#     #                                eta_yt=5e-4,
#     #                                eta_xtkm=5e-4) # this seems very slow? 2hr!!!! 
    
#     # GC_vectors4 = causal_block_flow(myVid4, pcca_cause_block, 
#     #                                block_size=3,
#     #                                k=1, 
#     #                                m=3, 
#     #                                eta_xt=5e-4, 
#     #                                eta_yt=5e-4,
#     #                                eta_xtkm=5e-4) # this seems very slow? 2hr!!!! 
    
#     # GC_vectors8 = causal_block_flow(myVid8, pcca_cause_block, 
#     #                                block_size=3,
#     #                                k=1, 
#     #                                m=3, 
#     #                                eta_xt=5e-4, 
#     #                                eta_yt=5e-4,
#     #                                eta_xtkm=5e-4) # this seems very slow? 2hr!!!! 
#     GC_vectors = GC_vectors 
#     GC_vectors2_resize = np.dstack([sktform.resize(GC_vectors2[...,ch], output_shape=myVid.shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
#     GC_vectors4_resize = np.dstack([sktform.resize(GC_vectors4[...,ch], output_shape=myVid.shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
#     GC_vectors8_resize = np.dstack([sktform.resize(GC_vectors8[...,ch], output_shape=myVid.shape[1:], preserve_range=True, order=1) for ch in np.arange(2)])
    GC_vectors = GC_vectors 
    GC_vectors2_resize = sktform.resize(GC_vectors2, output_shape=myVid.shape[1:]+GC_vectors2.shape[-2:], preserve_range=True, order=1)
    GC_vectors4_resize = sktform.resize(GC_vectors4, output_shape=myVid.shape[1:]+GC_vectors4.shape[-2:], preserve_range=True, order=1)
    GC_vectors8_resize = sktform.resize(GC_vectors8, output_shape=myVid.shape[1:]+GC_vectors8.shape[-2:], preserve_range=True, order=1)
#     # what is the best way to combine? 
#     # GC_vectors_combine = 1./4*(GC_vectors + GC_vectors2_resize + GC_vectors4_resize + GC_vectors8_resize)
#     # GC_vectors_combine = 1*GC_vectors + 2*GC_vectors2_resize +4* GC_vectors4_resize + 8*GC_vectors8_resize
#     # GC_vectors_combine = GC_vectors_combine / ( 1 + 2 + 4 + 8)
    
# # =============================================================================
# #     2. is this really the best way to combine ? 
# # =============================================================================
    GC_vectors_combine = np.nanmean( np.array([GC_vectors, GC_vectors2_resize, GC_vectors4_resize, GC_vectors8_resize]), axis=0) # combine 
#     # GC_vectors_combine = np.nanmean( np.array([GC_vectors2_resize, GC_vectors4_resize, GC_vectors8_resize]), axis=0)
    
    # # collapse the combine all into one into vectors.
    # mid = GC_vectors_combine.shape[-1]//2
            
    # corr_x_direction = -np.apply_over_axes(np.nansum, GC_vectors_combine[:,:,:,:mid], [-1,-2]) + np.apply_over_axes(np.nansum, GC_vectors_combine[:,:,:,mid+1:], [-1,-2])
    # corr_y_direction = -np.apply_over_axes(np.nansum, GC_vectors_combine[:,:,:mid], [-1,-2]) + np.apply_over_axes(np.nansum, GC_vectors_combine[:,:,mid+1:], [-1,-2])
    # intensity = np.nansum(GC_vectors_combine.reshape(GC_vectors_combine.shape[0], GC_vectors_combine.shape[1], -1), axis=-1) #* np.sqrt(corr_x_direction**2 + corr_y_direction**2)
    
    # mean_vector = np.array([np.squeeze(corr_y_direction), 
    #                          np.squeeze(corr_x_direction)])
    # mean_vector = mean_vector * intensity[None,...]
    # mean_vector = mean_vector.transpose(1,2,0)
    # mean_vector = -mean_vector
    
    mean_vector = causal_block_flow_scores_gradient(GC_vectors_combine)


    xy_coords = np.indices(myVid.shape[1:]); xy_coords=xy_coords.transpose(1,2,0)
    xy_coords = xy_coords[...,::-1]
    
    sampling = 8
#     plt.figure(figsize=(15,15))
#     plt.imshow(myVid[1])
#     plt.quiver(xy_coords[::sampling,::sampling,0], 
#                 xy_coords[::sampling,::sampling,1], 
#                 GC_vectors[::sampling,::sampling,1],  # x 
#                 -GC_vectors[::sampling,::sampling,0]) # y 
#     plt.show()
    
    
#     plt.figure(figsize=(15,15))
#     plt.title('2x')
#     plt.imshow(myVid[1])
#     plt.quiver(xy_coords[::sampling,::sampling,0], 
#                 xy_coords[::sampling,::sampling,1], 
#                 GC_vectors2_resize[::sampling,::sampling,1],  # x 
#                 -GC_vectors2_resize[::sampling,::sampling,0]) # y 
#     plt.show()
    
    
#     plt.figure(figsize=(15,15))
#     plt.title('4x')
#     plt.imshow(myVid[1])
#     plt.quiver(xy_coords[::sampling,::sampling,0], 
#                 xy_coords[::sampling,::sampling,1], 
#                 GC_vectors4_resize[::sampling,::sampling,1],  # x 
#                 -GC_vectors4_resize[::sampling,::sampling,0]) # y 
#     plt.show()
    
#     plt.figure(figsize=(15,15))
#     plt.title('8x')
#     plt.imshow(myVid[1])
#     plt.quiver(xy_coords[::sampling,::sampling,0], 
#                 xy_coords[::sampling,::sampling,1], 
#                 GC_vectors8_resize[::sampling,::sampling,1],  # x 
#                 -GC_vectors8_resize[::sampling,::sampling,0]) # y 
#     plt.show()
    
#     plt.figure(figsize=(15,15))
#     plt.title('Combine')
#     plt.imshow(myVid[1])
#     plt.quiver(xy_coords[::sampling,::sampling,0], 
#                 xy_coords[::sampling,::sampling,1], 
#                 GC_vectors_combine[::sampling,::sampling,1],  # x 
#                 -GC_vectors_combine[::sampling,::sampling,0]) # y 
#     plt.show()


    plt.figure(figsize=(15,15))
    plt.title('Combine')
    plt.imshow(myVid[1])
    plt.quiver(xy_coords[::sampling,::sampling,0], 
                xy_coords[::sampling,::sampling,1], 
                mean_vector[::sampling,::sampling,1],  # x 
                -mean_vector[::sampling,::sampling,0]) # y 
    plt.show()
    
    
#     # plt.figure(figsize=(15,15))
#     # # plt.imshow(myVid[1])
#     # plt.imshow(frame_a_[...,0])
#     # plt.plot(x.ravel(), 
#     #          y.ravel(), 'k.')
#     # plt.show()
    
    
    
    mean_flow_color = flow_to_color(mean_vector[...,::-1])
    
    plt.figure(figsize=(5,5))
    plt.imshow(mean_flow_color)
    plt.show()
    
    
    # """
    # This is correct!..... 
    # """
    # optical_flow_params = dict(pyr_scale=0.5, levels=1, winsize=3, iterations=5, poly_n=3, poly_sigma=1.2, flags=0)
     
    # vid_flow = extract_optflow(255*myVid[:], 
    #                             optical_flow_params, 
    #                             rescale_intensity=False, 
    #                             intensity_range=[2,98])
    
    # mean_opt_flow_color = flow_to_color(vid_flow.mean(axis=0))
    
    # plt.figure(figsize=(5,5))
    # plt.imshow(mean_opt_flow_color)
    # plt.show()
    
    
    optical_flow_params = dict(pyr_scale=0.5, levels=4, winsize=5, iterations=5, poly_n=3, poly_sigma=1.2, flags=0)
     
    vid_flow = extract_optflow(255*myVid[:], 
                                optical_flow_params, 
                                rescale_intensity=False, 
                                intensity_range=[2,98])
    
    mean_opt_flow_color = flow_to_color(vid_flow.mean(axis=0))
    
    plt.figure(figsize=(5,5))
    plt.imshow(mean_opt_flow_color)
    plt.show()
    
    
    plt.figure(figsize=(15,15))
    plt.title('mean optical flow')
    plt.imshow(myVid[1])
    plt.quiver(xy_coords[::sampling,::sampling,0], 
               xy_coords[::sampling,::sampling,1], 
                np.nanmean(vid_flow, axis=0)[::sampling,::sampling,0],  # x 
                -np.nanmean(vid_flow, axis=0)[::sampling,::sampling,1]) # y 
    plt.show()
    
    
    plt.figure(figsize=(5,5))
    plt.subplot(121)
    plt.title('Optical flow')
    plt.imshow(mean_opt_flow_color)
    plt.subplot(122)
    plt.title('linear GC')
    plt.imshow(mean_flow_color)
    plt.show()
    
    
    
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


