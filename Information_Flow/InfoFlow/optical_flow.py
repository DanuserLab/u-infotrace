# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 01:20:48 2022

@author: fyz11
"""


def Eval_dense_optic_flow(prev, present, params):
    r""" Computes the optical flow using Farnebacks Method

    Parameters
    ----------
    prev : numpy array
        previous frame, m x n image
    present :  numpy array
        current frame, m x n image
    params : Python dict
        a dict object to pass all algorithm parameters. Fields are the same as that in the opencv documentation, https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html. Our recommended starting values:
                
            * params['pyr_scale'] = 0.5
            * params['levels'] = 3
            * params['winsize'] = 15
            * params['iterations'] = 3
            * params['poly_n'] = 5
            * params['poly_sigma'] = 1.2
            * params['flags'] = 0
        
    Returns
    -------
    flow : finds the displacement field between frames, prev and present such that :math:`\mathrm{prev}(y,x) = \mathrm{next}(y+\mathrm{flow}(y,x)[1], x+\mathrm{flow}(y,x)[0])` where (x,y) is the cartesian coordinates of the image.
    """
    
    import numpy as np 
    import warnings
    import cv2

    # Check version of opencv installed, if not 3.0.0 then issue alert.
#    if '3.0.0' in cv2.__version__ or '3.1.0' in cv2.__version__:
        # Make the image pixels into floats.
    prev = prev.astype(np.float)
    present = present.astype(np.float)

    if cv2.__version__.split('.')[0] == '3' or cv2.__version__.split('.')[0] == '4':
        flow = cv2.calcOpticalFlowFarneback(prev, present, None, params['pyr_scale'], params['levels'], params['winsize'], params['iterations'], params['poly_n'], params['poly_sigma'], params['flags']) 
    if cv2.__version__.split('.')[0] == '2':
        flow = cv2.calcOpticalFlowFarneback(prev, present, pyr_scale=params['pyr_scale'], levels=params['levels'], winsize=params['winsize'], iterations=params['iterations'], poly_n=params['poly_n'], poly_sigma=params['poly_sigma'], flags=params['flags']) 
#    print(flow.shape)
    return flow


def rescale_intensity_percent(img, intensity_range=[2,98]):

    from skimage.exposure import rescale_intensity
    import numpy as np 

    p2, p98 = np.percentile(img, intensity_range)
    img_ = rescale_intensity(img, in_range=(p2,p98))

    return img_

# add in optical flow. 
def extract_optflow(vid, optical_flow_params, rescale_intensity=True, intensity_range=[2,98]): 
    # uses CV2 built in farneback ver. which is very fast and good for very noisy and small motion
    import cv2
    from skimage.exposure import rescale_intensity
    import numpy as np
    from tqdm import tqdm 

    vid_flow = []
    n_frames = len(vid)

    for frame in tqdm(np.arange(len(vid)-1)):
        frame0 = rescale_intensity_percent(vid[frame], intensity_range=intensity_range)
        frame1 = rescale_intensity_percent(vid[frame+1], intensity_range=intensity_range)
        flow01 = Eval_dense_optic_flow(frame0, frame1, 
                                       params=optical_flow_params)
        vid_flow.append(flow01)
    vid_flow = np.array(vid_flow).astype(np.float32) # to save some space. 

    return vid_flow