import os
import re
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.filters as flt
import statistics as stats

from skimage.io import imread
from skimage.io import imsave
from skimage.io import imread_collection
from skimage.feature import peak_local_max
from skimage.segmentation import random_walker
from skimage.segmentation import watershed

'''Spot counter function'''
def spot_counter(image, block_size, threshold, min_distance):
    # Set thresholds to reduce noise
    thresh_local = flt.threshold_local(image, block_size, offset=5) 
    # Find and label seeds
    seeds = peak_local_max(thresh_local, threshold_abs=threshold, indices=False, min_distance=min_distance)
    seeds_dil = ndi.filters.maximum_filter(seeds, size=2)
    seeds_masked = np.ma.array(seeds_dil, mask=seeds_dil==0)
    seeds_labeled = ndi.label(seeds)[0]
    seeds_labeled_dil = ndi.filters.maximum_filter(seeds_labeled, size=2)
    seeds_unique = len(np.unique(seeds_labeled_dil)[1:])
    return seeds_unique

'''Workflow function'''
def run_pipeline(root, outfolder):
    img_col_0 = np.zeros((1002, 1004))
    img_col_1 = np.zeros((1002, 1004))

    # Import all images in root
    for file in os.listdir(root):
        if not file.endswith('.tif'):
            continue
        _img = imread(os.path.join(root, file))
        if 'c2' in file: # DAPI
            img_col_0 = np.dstack((img_col_0, _img))
        if 'c1' in file: # Granule
            img_col_1 = np.dstack((img_col_1, _img))

    # Maximum projection and stacking of images
    img0 = np.max(img_col_0, axis=2)
    img1 = np.max(img_col_1, axis=2)
    img = np.stack((img0, img1), axis=0)
    img = img.astype(np.uint16)

    # Segment file name
    filename = os.path.basename(os.path.normpath(root))
    filename_cell = re.search(r'^(.*?)[CR]', filename).group(0)
    filename_cell, filename_transgene = filename_cell.split('-')
    filename_condition = re.findall(r'\-(..)\-', filename)[0]
    filename_out = os.path.join(outfolder, ''.join([filename, '.csv']))
    os.makedirs(outfolder, exist_ok=True)

    # Gauss-smoothen image
    img0_sigma = 3
    img0_smooth = ndi.filters.gaussian_filter(img[0], img0_sigma)

    # Decide on a thresholding option, here: otsu
    img0_thresh = flt.threshold_otsu(img0_smooth)
    img0_mem = img0_smooth > img0_thresh

    # Labeling of thresholded DAPI signal, smoothing
    img0_cell_labels, _ = ndi.label(img0_mem)
    img0_dist_trans = ndi.distance_transform_edt(img0_mem)
    img0_dist_trans_smooth = ndi.filters.gaussian_filter(img0_dist_trans, sigma=3)

    # Find local maxima – label seeds
    img0_seeds = peak_local_max(img0_dist_trans_smooth, indices=False, min_distance=5)
    img0_seeds_dil = ndi.filters.maximum_filter(img0_seeds, size=10) # Dilates seeds making them more visible
    img0_seeds_labeled = ndi.label(img0_seeds)[0]
    img0_seeds_labeled_dil = ndi.filters.maximum_filter(img0_seeds_labeled, size=10)

    # Use random_walker algorithm (inverted) to segment cells
    img0_seg = watershed(~img0_smooth, img0_seeds_labeled)

    # Create image border mask
    img0_seg_clean = np.copy(img0_seg)
    img0_border_mask = np.zeros(img0_seg.shape, dtype=np.bool)
    img0_border_mask = ndi.binary_dilation(img0_border_mask, border_value=1)

    # Delete border cells and re-label the remainder
    for cell_ID in np.unique(img0_seg):
        img0_cell_mask = img0_seg==cell_ID 
        img0_cell_border_overlap = np.logical_and(img0_cell_mask, img0_border_mask)
        img0_total_overlap_pixels = np.sum(img0_cell_border_overlap)
        if img0_total_overlap_pixels > 0: 
            img0_seg_clean[img0_cell_mask] = 0
    for new_ID, cell_ID in enumerate(np.unique(img0_seg_clean)[1:]):
        img0_seg_clean[img0_seg_clean==cell_ID] = new_ID + 1   
        
    # Parameters for detection
    img1 = img[1]
    img1_sigma_all = [0, 1]
    img1_block_size = 5
    img1_thresh_abs  = [8000, 9000, 10000, 11000]
    img1_min_distance_all = [0, 5]
    img1_thresh_area = 3000

    results = {'filename' : [],
            'sigma' : [],
            'thresh_local' : [],
            'block_size' : [],
            'min_distance' : [],
            'cell_line' : [],
            'transgene' : [],
            'condition' : [],
            'cell_ID' : [],
            'cell_area' : [],
            'cell_intensity_mean': [],
            'cell_intensity_std' : [],
            'spots'     : []}

    for img1_sigma in img1_sigma_all:
        for img1_thresh in img1_thresh_abs:
            for img1_min_distance in img1_min_distance_all:
                img1_smooth = ndi.filters.gaussian_filter(img[1], img1_sigma)

                for cell_ID in np.unique(img0_seg_clean)[1:]:
                    img0_cell_mask = img0_seg_clean==cell_ID
                    img1_cell = np.where(np.ma.array(img0_cell_mask, mask=img0_cell_mask==0), img1_smooth, 0)
                    img1_cell_area = np.count_nonzero(img1_cell > img1_thresh_area)
                    img1_cell_intensity_mean = np.mean(np.nonzero(img1_cell))
                    img1_cell_intensity_std = np.std(np.nonzero(img1_cell))
                    img1_cell_granules = spot_counter(img1_cell, img1_block_size, img1_thresh, img1_min_distance)

                    results['filename'].append(os.path.basename(os.path.normpath(root)))
                    results['cell_line'].append(filename_cell)
                    results['transgene'].append(filename_transgene)
                    results['condition'].append(filename_condition)
                    results['sigma'].append(img1_sigma)
                    results['thresh_local'].append(img1_thresh)
                    results['block_size'].append(img1_block_size)
                    results['min_distance'].append(img1_min_distance)
                    results['cell_ID'].append(cell_ID)
                    results['cell_area'].append(img1_cell_area)
                    results['cell_intensity_mean'].append(int(img1_cell_intensity_mean))
                    results['cell_intensity_std'].append(int(img1_cell_intensity_std))
                    results['spots'].append(img1_cell_granules)

    # Save results in a csv file
    with open(filename_out,'w') as outfile:
        header_string = '\t'.join(results.keys()) + '\n'
        outfile.write(header_string)
        for index in range(len(results['cell_ID'])):
            data_string = '\t'.join([str(results[key][index]) for key in results.keys()]) + '\n'
            outfile.write(data_string)
