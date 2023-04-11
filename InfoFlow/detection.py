# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 17:10:21 2022

@author: fyz11
"""

def binary_segment_magnitude_flow(vector_img, min_size=100):
    
    import numpy as np 
    import skimage.filters as skfilters
    import skimage.morphology as skmorph
    
    
    mag = np.linalg.norm(vector_img, axis=-1)
    # binary = mag >= skfilters.threshold_otsu(mag)
    binary = mag >= np.mean(mag) + np.nanstd(mag)
    binary = skmorph.remove_small_objects(binary, min_size=min_size)
    
    return binary 


def detect_bbox_objects_from_flow(vector_img, prob_img=None, min_size=100, connectivity=1):
    
    import numpy as np 
    
    binary = binary_segment_magnitude_flow(vector_img, min_size=min_size)
    if prob_img is None:
        mag_map = np.linalg.norm(vector_img, axis=-1)
    else:
        mag_map = prob_img.copy()
    
    binary_bboxes = extract_bboxes_binary(binary, prob=mag_map, connectivity=connectivity)
    
    return binary, binary_bboxes


def largest_component_area(binary, connectivity=1):
    
    from skimage.measure import label, regionprops
    import numpy as np 
    
    binary_labelled = label(binary, connectivity=connectivity)
    # largest component.
    binary_props = regionprops(binary_labelled)
    binary_vols = [re.area for re in binary_props]
    binary_out = binary_labelled == (np.unique(binary_labelled)[1:][np.argmax(binary_vols)])
    
    return binary_out

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    import numpy as np 
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_bbox_binary_2D(mask, prob=None):
    
    # score each binary using integrated softmax sums if prob is passed!.
    
    import numpy as np 
    yyxx = np.argwhere(mask>0) 
    yy = yyxx[:,0]
    xx = yyxx[:,1]
    
    x1 = np.min(xx)
    x2 = np.max(xx)
    y1 = np.min(yy)
    y2 = np.max(yy)
    
    import pylab as plt 
    plt.figure()
    plt.imshow(mask)
    plt.show()

    plt.figure()
    plt.imshow(prob*(mask>0)*1.)
    plt.show()

    if x2>x1 and y2>y1: 
        # this is to ensure it is valid. 
        if prob is not None:
            # how to score the probs!. 
            prob_crop = prob[int(y1):int(y2)+1, int(x1):int(x2)].copy()
            mask_crop = mask[int(y1):int(y2)+1, int(x1):int(x2)].copy()
            
            ones = np.nansum(prob_crop*mask_crop) / (np.nansum(mask_crop))
            zeros = np.nansum(prob_crop*(1-mask_crop)) / (np.nansum(1-mask_crop))            
            p = softmax([ones,zeros])

            bbox = np.hstack([p[0], x1,y1,x2,y2])
        else:
            bbox = np.hstack([x1,y1,x2,y2])
    else:
        bbox = []
        
    return bbox


def extract_bboxes_binary(binary, prob=None, connectivity=1):

    import numpy as np 
    import skimage.measure as skmeasure 
    
    bboxes = []
    
    labelled_mask = skmeasure.label(binary, connectivity=connectivity) # 
    uniq_regions = np.setdiff1d(np.unique(labelled_mask),0)
    
    for region in uniq_regions:
        
        mask_region = labelled_mask==region 
        bbox = get_bbox_binary_2D(mask_region, prob=prob)
        if len(bbox) > 0: 
            bboxes.append(bbox)
            
    if len(bboxes) > 0:   
        return labelled_mask, uniq_regions, bboxes
    else:
        return []
    
    
    