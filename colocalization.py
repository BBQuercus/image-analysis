#!/usr/bin/env python
# coding: utf-8

import os
import re
import sys
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import skimage.filters as flt
import scipy.stats as st

from skimage.io import imread
from skimage.io import imsave
from scipy.signal import convolve2d
from matplotlib.patches import Patch
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects

def run_pipeline(files, root, outfolder):
    '''
    Runs a colocalization and spot detection pipeline.
    Parameters
    ----------
    files: list of str
        List of unique image files including route.
    root: dir
        Directory where image files are located.
    outfolder: dir
        Directory for output.
    Returns
    ----------
    outfolder/index.csv: .csv file
        File with all values in the outfolder.
    '''
    # Wavelengths to analyze and resulting files
    lambdas = [405, 561, 640]

    # Find sharpest image in img0_lambda channel
    for f in files:
        if str(lambdas[0]) in f:
            img_sharp = sharp(imread(f))

    # Open all images corresponding to a lambda
    for f in files:
        for i in range(len(lambdas)):
            if str(lambdas[i]) in f:
                globals()['img_'+str(lambdas[i])] = imread(f)[img_sharp]

    # Stack and convert to uint16
    img = np.stack([globals()['img_'+str(lambdas[i])] for i in range(len(lambdas))], axis=0)

    # Parameters for cellular segmentation
    img0_sigma = 3
    img0_min_size = 200
    img0_min_distance = 50
    img0_dilation = 5
    img0_thresh_bg = 1000
    img0_min_size_bg = 10_000

    # Gauss-smoothening
    img0_smooth = ndi.filters.gaussian_filter(img[0], img0_sigma)

    # Thresholding and removal of small objects
    img0_thresh = flt.threshold_otsu(img0_smooth)
    img0_smooth_thresh = img0_smooth>img0_thresh
    img0_mem = remove_small_objects(img0_smooth_thresh, img0_min_size)

    # Labeling, euclidean distance transform, smoothing
    img0_cell_labels, _ = ndi.label(img0_mem)
    img0_dist_trans = ndi.distance_transform_edt(img0_mem)
    img0_dist_trans_smooth = ndi.filters.gaussian_filter(img0_dist_trans, sigma=img0_sigma)

    # Seeding (and dilating for visualization)
    img0_seeds = peak_local_max(img0_dist_trans_smooth, indices=False, min_distance=img0_min_distance)
    img0_seeds_labeled = ndi.label(img0_seeds)[0]
    img0_seeds_labeled_dil = ndi.filters.maximum_filter(img0_seeds_labeled, size=10)

    # Inverted watershed for segmentation
    img0_seg = watershed(~img0_smooth, img0_seeds_labeled)

    # Removal of dilated nuclei
    img0_seg_nuclei = img0_seg * ~convolve2d(img0_mem.astype(int), dilate(img0_dilation).astype(int), mode='same').astype(bool)

    # Removal of background based on colocalization channel 1
    img0_thresh_bg = flt.threshold_triangle(img[1])
    img0_smooth_thresh_bg = img[1]>img0_thresh_bg
    img0_seg_clean = img0_seg_nuclei * img0_smooth_thresh_bg

    # Removal of 'debris'
    img0_seg_labeled, img0_seg_labels = ndi.measurements.label(img0_seg_clean)
    img0_seg_label_size = [(img0_seg_labeled == label).sum() for label in range(img0_seg_labels + 1)]
    for label, size in enumerate(img0_seg_label_size):
        if size < img0_min_size_bg:
            img0_seg_clean[img0_seg_labeled == label] = 0
    
    # Parameters for spot detection and colocalization
    img1 = img[1]
    img1_sigma = 1
    img1_block_size = 5
    img1_min_distance = 0
    img1_thresh = 5000

    img2 = img[2]
    img2_sigma = 1
    img2_block_size = 5
    img2_min_distance = 0
    img2_thresh = 5000

    psf_lambda = 648
    psf_na = 1.45
    psf = int((0.6 * psf_lambda) / psf_na)
    scrambles = 1000

    results = {'filename'                 : [],
               'cell_ID'                  : [],
               
               # Measurements
               'cell_area'                : [],
               'img1_cell_intensity_mean' : [],
               'img1_cell_intensity_std'  : [],
               'img1_spot_count'          : [],
               'img2_cell_intensity_mean' : [],
               'img2_cell_intensity_std'  : [],
               'img2_spot_count'          : [],
               'img1_2_coloc_org'         : [],
               'img1_2_coloc_scr'         : [],
               
               # Parameters
               'psf'                      : [],
               'scrambles'                : [],
               'img0_sigma'               : [],
               'img0_min_size'            : [],
               'img0_min_distance'        : [],
               'img0_dilation'            : [],
               'img0_thresh_bg'           : [],
               'img0_min_size_bg'         : [],
               'img1_sigma'               : [],
               'img1_block_size'          : [],
               'img1_min_distance'        : [],
               'img1_thresh'              : [],
               'img2_sigma'               : [],
               'img2_block_size'          : [],
               'img2_min_distance'        : [],
               'img2_thresh'              : []}
               
    # Smoothing for spot detection
    img1_smooth = ndi.filters.gaussian_filter(img1, img1_sigma)
    img2_smooth = ndi.filters.gaussian_filter(img2, img2_sigma)

    for cell_ID in np.unique(img0_seg_clean)[1:]:
        img0_cell_mask = img0_seg_clean==cell_ID
        
        # Area measurement on cell mask
        cell_area = np.count_nonzero(img0_cell_mask)
        
        # Intensity and spot count measurements
        img1_cell = np.where(np.ma.array(img0_cell_mask, mask=img0_cell_mask==0), img1_smooth, 0)
        img1_cell_intensity_mean = np.mean(np.nonzero(img1_cell))
        img1_cell_intensity_std = np.std(np.nonzero(img1_cell))
        img1_spot_count = spot_counter(img1_cell, img1_block_size, img1_thresh, img1_min_distance)
        img2_cell = np.where(np.ma.array(img0_cell_mask, mask=img0_cell_mask==0), img2_smooth, 0)
        img2_cell_intensity_mean = np.mean(np.nonzero(img2_cell))
        img2_cell_intensity_std = np.std(np.nonzero(img2_cell))
        img2_spot_count = spot_counter(img2_cell, img2_block_size, img2_thresh, img2_min_distance)
        
        # Colocalization measurements       
        img1_2_coloc_scr, img1_2_coloc_org = coloc((img1*img0_cell_mask), (img2*img0_cell_mask), psf, scrambles)
            
        results['filename'].append(os.path.basename(os.path.normpath(root)))
        results['cell_ID'].append(cell_ID)
        
        # Measurements
        results['cell_area'].append(cell_area)
        results['img1_cell_intensity_mean'].append(img1_cell_intensity_mean)
        results['img1_cell_intensity_std'].append(img1_cell_intensity_std)
        results['img1_spot_count'].append(img1_spot_count)
        results['img2_cell_intensity_mean'].append(img2_cell_intensity_mean)
        results['img2_cell_intensity_std'].append(img2_cell_intensity_std)
        results['img2_spot_count'].append(img2_spot_count)
        results['img1_2_coloc_org'].append(img1_2_coloc_org)
        results['img1_2_coloc_scr'].append(img1_2_coloc_scr)
        
        # Parameters
        results['psf'].append(psf)
        results['scrambles'].append(scrambles)
        results['img0_sigma'].append(img0_sigma)
        results['img0_min_size'].append(img0_min_size)
        results['img0_min_distance'].append(img0_min_distance)
        results['img0_dilation'].append(img0_dilation)
        results['img0_thresh_bg'].append(img0_thresh_bg)
        results['img0_min_size_bg'].append(img0_min_size_bg)
        results['img1_sigma'].append(img1_sigma)
        results['img1_block_size'].append(img1_block_size)
        results['img1_min_distance'].append(img1_min_distance)
        results['img1_thresh'].append(img1_thresh)
        results['img2_sigma'].append(img2_sigma)
        results['img2_block_size'].append(img2_block_size)
        results['img2_min_distance'].append(img2_min_distance)
        results['img2_thresh'].append(img2_thresh)

    # Export results
    with open(''.join([files[0], '.csv']),'w') as outfile:
        header_string = '\t'.join(results.keys()) + '\n'
        outfile.write(header_string)
        for index in range(len(results['filename'])):
            data_string = '\t'.join([str(results[key][index]) for key in results.keys()]) + '\n'
            outfile.write(data_string)

def sharp(img):
    '''
    Returns index of the sharpest slice in an image array of shape z, x, y.
    Parameters
    ----------
    img: np.array
        Image array to determine sharpness.
    Returns
    ----------
    output: float
        Sharpness of array.
    '''
    sharpness = []
    array = np.asarray(img, dtype=np.int32)
    for i in range(array.shape[0]):
        y, x = np.gradient(array[i])
        norm = np.sqrt(x**2 + y**2)
        sharpness.append(np.average(norm))
    return sharpness.index(max(sharpness))

def dilate(dil=0):
    '''
    Given a int, creates a kernel of that size.
    Parameters
    ----------
    dil: int, default 0
        Size of kernel.
    Returns
    ----------
    output: np.array(dtype=bool)
        Kernel of size dil.
    '''
    a = []
    b = []
    for i in range(dil):
        a.append(['F'*(dil-i)+'T'*(i+1)+'T'*(i)+'F'*(dil-i)])
    a.append(('T'*(dil*2+1)).split())
    for i in range(dil-1, -1, -1):
        a.append(['F'*(dil-i)+'T'*(i+1)+'T'*(i)+'F'*(dil-i)])
    for i in a:
        for j in i:
            b.append(list(j))
    for n,i in enumerate(b): 
        for m,j in enumerate(i): 
            if j == 'F': 
                b[n][m] = False 
            if j == 'T': 
                b[n][m] = True 
    return np.array(b, dtype=bool)

def mirror_edges(img, psf_width):
    '''
    Given a 2D image, boundaries are padded by mirroring so that
    the dimensions of the image are multiples for the width of the
    point spread function.
    Parameters
    ----------
    im: array_like
        Image to mirror edges.
    psf_width: int
        The width, in pixels, of the point spread function.
    Returns
    ----------
    output: array_like
        Image with mirrored edges.
    '''
    # Required padding
    pad_i = psf_width - (img.shape[0] % psf_width)
    pad_j = psf_width - (img.shape[1] % psf_width)
    
    # Width of padding
    pad_top = pad_i // 2
    pad_bot = pad_i - pad_top
    pad_left = pad_j // 2
    pad_right = pad_j = pad_left
    
    # Numpy padding
    return np.pad(img, ((pad_top, pad_bot), (pad_left, pad_right)), mode='reflect')

def img_to_blocks(img, width, roi=None, roi_method='all'):
    '''
    Converts image to list of square subimages called 'blocks'.
    Parameters
    ----------
    im: array_like
        Image to convert to a list of blocks.
    width: int
        Width of square blocks in pixels.
    roi: array_like, dtype bool, default None
        Boolean image the same shape as 'im_1' and 'im_2' that
        is True for pixels within the ROI.
    roi_method: str, default 'all'
        If 'all', all pixels of a given subimage must be within
        the ROI for the subimage itself to be considered part
        of the ROI.
        If 'any', if any one pixel is within the ROI,
        the subimage is considered part of the ROI.
    Returns
    ----------
    output: list of ndarrays
        Each entry is a 'width' by 'width' Numpy array containing
        a block.
    '''
    # ROI initialization
    if roi is None:
        roi = np.ones_like(img)
    
    # Method for determining if in ROI or not
    if roi_method == 'all':
        roi_test = np.all
    else:
        roi_test = np.any
        
    # Construction of block list
    return np.array([img[i:i + width, j:j + width]
                        for i in range(0, img.shape[0], width)
                            for j in range(0, img.shape[1], width)
                                if roi_test(roi[i:i + width, j:j + width])])

def scramble(blocks_1, blocks_2_flat, scrambles):
    '''
    Scrambles blocks_1 n_scramble times and returns the Pearson r values.
    Parameters
    ----------
    blocks_1: np.array
        Array to to be scrambled values.
    blocks_2_flat: array
        Array of values staying constant.
    scrambles: int
        Number of scrambles.
    Returns
    ----------
    output: list of floats
        Pearson r values.
    '''
    r_scr = np.zeros(scrambles)
    for i in range(scrambles):
        random.shuffle(blocks_1)
        r, _ = st.pearsonr(np.array(blocks_1).ravel(), blocks_2_flat)
        r_scr[i] = r
    r_scr = [i for i in r_scr if ~np.isnan(i)]
    return r_scr

def coloc(img1, img2, psf=3, scrambles=1000):
    '''
    Colocalization of two images and comparison with scrambles.
    Parameters
    ----------
    img1: nd.array
        Image array of to be scrambled image.
    img2: nd.array
        Image array of the 'static' image.
    psf: int, default 3
        Point spread function of the longer wavelength.
    scrambles: int, default 1000
        Number of scrambles to calculate r value of.
    Returns
    ----------
    outputs: float, list of floats
        R values (unscrambled (float) and scrambled(list of floats))
    '''
    # Mirror edges
    img1_mirror = mirror_edges(img1, psf)
    img2_mirror = mirror_edges(img2, psf)

    # Generate blocks of both channels
    img1_blocks = img_to_blocks(img1_mirror, psf)
    img2_blocks = img_to_blocks(img2_mirror, psf)

    # Store blocks of channel 2 as flattened array (not scrambled)
    img2_blocks_flat = np.array(img2_blocks).flatten()

    # Scamblin' and obtain R value
    img1_scr = scramble(img1_blocks, img2_blocks_flat, scrambles)
    
    # Unscrambled R value
    img1_unscr, p = st.pearsonr(np.array(img1_blocks).ravel(), img2_blocks_flat)
    
    return img1_unscr, img1_scr

def spot_counter(img_, block_size=1, threshold=0, min_distance=0):
    '''
    Counts spots returning the number of unique spots in a given area.
    Parameters
    ----------
    img_: np.array.shape(x, x)
        Image on which spots should be counted.
    block_size: int, default 0
         
    threshold: int, default 0
        Local maximum threshold filtering out any lower values.
    min_distance: int, default 0
        Minimum distance between two potential spots.
    Returns
    ----------
    output: int
        Unique seed count.
    '''
    thresh = flt.threshold_local(img_, block_size, offset=5) 
    seeds = peak_local_max(thresh, indices=False, min_distance=min_distance)
    seeds_labeled = ndi.label(seeds)[0]
    seeds_unique = len(np.unique(seeds_labeled)[1:]) # '0' Background not included
    return seeds_unique
