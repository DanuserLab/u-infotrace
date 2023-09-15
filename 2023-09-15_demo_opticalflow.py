

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

from tqdm import tqdm 

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
compute mean multiscale Farneback optical flow - uses the opencv library. 
"""
optical_flow_params = dict(pyr_scale=0.5, 
                           levels=5, 
                           winsize=15, 
                           iterations=5, 
                           poly_n=3, 
                           poly_sigma=1.2, 
                           flags=0)
 
# note 255 is to make 8bit range image. 
vid_flow = extract_optflow(255.*myVid[:], 
                            optical_flow_params, 
                            rescale_intensity=False, 
                            intensity_range=[2,98])

optflow_vector = vid_flow.mean(axis=0)
mean_opt_flow_color = flow_to_color(vid_flow.mean(axis=0))


xy_coords = np.indices(myVid.shape[1:]); 
xy_coords=xy_coords.transpose(1,2,0)
xy_coords = xy_coords[...,::-1]

sampling = 4

plt.figure(figsize=(15,15))
plt.title('Multiscale Farneback Optical Flow')
plt.imshow(myVid[0])
plt.quiver(xy_coords[::sampling,::sampling,0], 
            xy_coords[::sampling,::sampling,1], 
            optflow_vector[::sampling,::sampling,0],  # x 
            -optflow_vector[::sampling,::sampling,1]) # y 
plt.show()

plt.figure(figsize=(15,15))
plt.title('optical flow baseline_flow-color')
plt.imshow(myVid[0], cmap='gray')
plt.imshow(mean_opt_flow_color)
plt.show()



"""
optical flow example using TVL1 flow (more smoother, more global with interpolation across non-moving regions)
"""
from InfoFlow.optical_flow import extract_optflow_TVL1 

optical_flow_params = dict(attachment=15, 
                           tightness=0.3, 
                           num_warp=5, 
                           num_iter=10, 
                           tol=0.0001, 
                           prefilter=False)

vid_flow_TVL1 = extract_optflow_TVL1(myVid[:], 
                                optical_flow_params, 
                                rescale_intensity=False, 
                                intensity_range=[2,98])


optflow_vector = vid_flow_TVL1.mean(axis=0)
mean_opt_flow_color = flow_to_color(vid_flow_TVL1.mean(axis=0))


xy_coords = np.indices(myVid.shape[1:]); 
xy_coords=xy_coords.transpose(1,2,0)
xy_coords = xy_coords[...,::-1]

sampling = 12

plt.figure(figsize=(15,15))
plt.title('TV-L1 Optical Flow')
plt.imshow(myVid[0])
plt.quiver(xy_coords[::sampling,::sampling,0], 
            xy_coords[::sampling,::sampling,1], 
            optflow_vector[::sampling,::sampling,0],  # x 
            -optflow_vector[::sampling,::sampling,1]) # y 
plt.show()

plt.figure(figsize=(15,15))
plt.title('TV-L1 optical flow baseline_flow-color')
plt.imshow(myVid[0], cmap='gray')
plt.imshow(mean_opt_flow_color)
plt.show()


