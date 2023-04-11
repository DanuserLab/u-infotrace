

# standard library imports
import numpy as np
import scipy.io
import scipy.io as spio 
import skimage.io as skio  
from scipy.ndimage import gaussian_filter
import pylab as plt # this is from matplotlib library

import skimage.transform as sktform
from skimage import color
# from tqdm import tqdm # this optional import is for monitoring progress
import skimage.transform as sktform
import skimage.util as skutil 
import scipy.ndimage as ndimage 

"""
Imports of various flow functions. 
"""
from InfoFlow.gc_flow import GC_full_reduced_separate_regress_individual
from InfoFlow.DDC_flow import DDC_cause
from InfoFlow.pdc_dtf_flow import PDC_central_flow
from InfoFlow.pcca_flow import pcca_cause_block
from InfoFlow.correlation_flow import nd_xcorr_lag

from InfoFlow.optical_flow import extract_optflow # this depends on opencv installation. please install with pip install opencv-contrib-python from https://pypi.org/project/opencv-contrib-python/
from InfoFlow.LK_flow import Linear_LK_cause
from InfoFlow.flow_vis import flow_to_color
from InfoFlow.utils import read_video_cv2

from InfoFlow.dynamic_image import _compute_dynamic_image
import InfoFlow.infoflow as infoflow_scripts


"""
1. read in the video file. 
"""
myVid = read_video_cv2(r'/archive/bioinformatics/Danuser_lab/shared/ComputerVisionDatasets/CrowdFlow_SaadAli/Data_Mov_Format/3687-18_70.mov')

myVid = color.rgb2gray(myVid) # convert to grayscale image


# a) debug visualization, computing the dynamic image. 
dyn_image_Vid = _compute_dynamic_image(myVid[...,None])

plt.figure(figsize=(5,5))
plt.title('Dynamic Image')
plt.imshow(dyn_image_Vid, cmap='coolwarm')
plt.show()


plt.figure(figsize=(5,5))
plt.title('Variance Image')
plt.imshow(np.var(myVid, axis=0), cmap='coolwarm')
plt.show()


"""
downscale the input image for speed and smooth at gaussian sigma 1. 
"""
myVid8 = infoflow_scripts.gaussian_video_pyramid(myVid, scales=[4], sigma=1)[0] # set to no smooth!. 

# =============================================================================
#     2. run through all causal measures and extract the output
# =============================================================================
causal_flow_outputs = [infoflow_scripts.causal_flow_scores(myVid8, GC_full_reduced_separate_regress_individual, winsize=3, lag=1, alpha=1),  # this is weird!. 
					   infoflow_scripts.causal_flow_scores(myVid8, DDC_cause, eps=1e-12, alpha=1e-2), 
					   infoflow_scripts.causal_flow_scores(myVid8, Linear_LK_cause, eps=1e-12),
					   infoflow_scripts.causal_flow_scores(myVid8, PDC_central_flow, lag=1, alpha=1e-2), 
                       infoflow_scripts.causal_flow_scores(myVid8, nd_xcorr_lag, lag=1)]

causal_flow_vectors = [infoflow_scripts.causal_block_flow_scores_gradient(flo) for flo in causal_flow_outputs] 
# 

# testing the PCCA block flow. 
pcca_flow = infoflow_scripts.causal_block_flow(myVid8, pcca_cause_block, 
                                   block_size=3,
                                   k=1, 
                                   m=1, 
                                   eta_xt=5e-4, 
                                   eta_yt=5e-4,
                                   eta_xtkm=5e-4)


# # =============================================================================
# #     3. Visualization of output. 
# # =============================================================================
xy_coords = np.indices(myVid8.shape[1:]); xy_coords=xy_coords.transpose(1,2,0)
xy_coords = xy_coords[...,::-1]

sampling = 1

flow_methods = ['cGC', 'DDC', 'LK', 'PDC', 'Corr']

for ii in np.arange(len(causal_flow_vectors)):
    plt.figure(figsize=(15,15))
    plt.title(flow_methods[ii])
    plt.imshow(myVid8[0])
    plt.quiver(xy_coords[::sampling,::sampling,0], 
                xy_coords[::sampling,::sampling,1], 
                causal_flow_vectors[ii][::sampling,::sampling,1],  # x 
                -causal_flow_vectors[ii][::sampling,::sampling,0]) # y 
    plt.show()

    
    mean_flow_color = flow_to_color(causal_flow_vectors[ii][...,::-1])
    
    plt.figure(figsize=(15,15))
    plt.title(flow_methods[ii]+'_flow-color')
    plt.imshow(myVid8[0], cmap='gray')
    plt.imshow(mean_flow_color, alpha=0.7)
    plt.show()


"""
Visualization for pcca flow vector 
"""

plt.figure(figsize=(15,15))
plt.title('PCCA')
plt.imshow(myVid8[0])
plt.quiver(xy_coords[::sampling,::sampling,0], 
            xy_coords[::sampling,::sampling,1], 
            pcca_flow[::sampling,::sampling,1],  # x 
            -pcca_flow[::sampling,::sampling,0]) # y 
plt.show()


mean_flow_color = flow_to_color(pcca_flow[...,::-1])

plt.figure(figsize=(15,15))
plt.title('PCCA_flow-color')
plt.imshow(myVid8[0], cmap='gray')
plt.imshow(mean_flow_color, alpha=0.7)
plt.show()


"""
Compare to optical flow - uses the opencv library. 
"""
optical_flow_params = dict(pyr_scale=0.5, levels=1, winsize=3, iterations=5, poly_n=3, poly_sigma=1.2, flags=0)
 
vid_flow = extract_optflow(255*myVid8[:], 
                            optical_flow_params, 
                            rescale_intensity=False, 
                            intensity_range=[2,98])

optflow_vector = vid_flow.mean(axis=0)
mean_opt_flow_color = flow_to_color(vid_flow.mean(axis=0))

plt.figure(figsize=(15,15))
plt.title('optical flow baseline')
plt.imshow(myVid8[0])
plt.quiver(xy_coords[::sampling,::sampling,0], 
            xy_coords[::sampling,::sampling,1], 
            optflow_vector[::sampling,::sampling,0],  # x 
            -optflow_vector[::sampling,::sampling,1]) # y 
plt.show()

plt.figure(figsize=(15,15))
plt.title('optical flow baseline_flow-color')
plt.imshow(myVid8[0], cmap='gray')
plt.imshow(mean_opt_flow_color)
plt.show()

