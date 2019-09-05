#!/usr/bin/env python
# coding: utf-8

import glob, os, random, re, sys
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import scipy.ndimage as ndi
import skimage.filters as flt
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from scipy.signal import convolve2d
from skimage.io import imread, imsave
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects


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
    a, b = [], []
    for i in range(dil):
        a.append(['0'*(dil-i)+'1'*(i+1)+'1'*(i)+'0'*(dil-i)])
    a.append(('1'*(dil*2+1)).split())
    for i in range(dil-1, -1, -1):
        a.append(['0'*(dil-i)+'1'*(i+1)+'1'*(i)+'0'*(dil-i)])
    for i in a:
        for j in i:
            b.append(list(j))
    return np.array(np.array(b, dtype=int), dtype=bool)


def coloc(img0, img1, img2, scrambles=1000):
    '''
    Colocalization of two images and comparison with scrambles.
    Parameters
    ----------
    img0: nd.array
        Mask array to 'overlay' over both images (e.g. cellular segmentation).
    img1: nd.array
        Image array of to be scrambled image.
    img2: nd.array
        Image array of the 'static' image.
    scrambles: int, default 1000
        Number of scrambles to calculate r value of.
    Returns
    ----------
    outputs: float, list of floats
        R values (unscrambled (float) and scrambled(list of floats))
    '''
    img1_seg = img1 * img0
    img2_seg = img2 * img0
    r_org, _ = st.pearsonr(img1_seg.flatten(), img2_seg.flatten())
    r_scr = []
    for _ in range(scrambles):
        img1_flat = img1.flatten()
        idx, _ = np.nonzero(img1_flat)
        img1_flat[idx] = img1_flat[np.random.permutation(idx)]
        img1_flat = img1_flat.reshape(img1_seg.shape)
        r, _ = st.pearsonr(img1_flat.flatten(), img2_seg.flatten())
        r_scr.append(r)
    return r_org, r_scr


def spot_counter(img_, block_size=1, threshold=0, min_distance=0):
    '''
    Counts spots returning the number of unique spots in a given area.
    Parameters
    ----------
    img_: np.array.shape(x, x)
        Image on which spots should be counted.
    block_size: int, default 0
        Odd size of pixel neighborhood to perform a local threshold.
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
    seeds_labeled, _ = ndi.label(seeds)
    seeds_unique = len(np.unique(seeds_labeled)[1:]) # '0' to not include background
    return seeds_unique


def run_pipeline(files, root, parameters):
    '''
    Runs a colocalization and spot detection pipeline.
    Parameters
    ----------
    index: str
        Base name / index for one set of image files.
    root: dir
        Directory where image files are located.
    parameters: dict    
        Dictionary with all required parameters (see parameter extraction below).
    Returns
    ----------
    root/index.csv: .csv file
        File with all values in the outfolder.
    '''
    # Parameter extraction
    img0_sigma = parameters['img0_sigma']
    img0_min_size = parameters['img0_min_size']
    img0_min_distance = parameters['img0_min_distance']
    img0_dilation = parameters['img0_dilation']
    img0_thresh_bg = parameters['img0_thresh_bg']
    img0_min_size_bg = parameters['img0_min_size_bg']
    img1_sigma = parameters['img1_sigma']
    img1_block_size = parameters['img1_block_size']
    img1_min_distance = parameters['img1_min_distance']
    img1_thresh = parameters['img1_thresh']
    img2_sigma = parameters['img2_sigma']
    img2_block_size = parameters['img2_block_size']
    img2_min_distance = parameters['img2_min_distance']
    img2_thresh = parameters['img2_thresh']
    scrambles = parameters['scrambles']

    lambdas = [405, 561, 640]

    # Find sharpest image in img2_lambda channel
    for f in files:
        if str(lambdas[2]) in f:
            img_sharp = sharp(imread(f))

    # Open all images corresponding to a lambda
    for f in files:
        for i in range(len(lambdas)):
            if str(lambdas[i]) in f:
                globals()['img_'+str(lambdas[i])] = imread(f)[img_sharp]

    # Stack and convert to uint16
    img = np.stack([globals()['img_'+str(lambdas[i])] for i in range(len(lambdas))], axis=0)
    img0 = img[0]
    img1 = img[1]
    img2 = img[2]

    # Gauss-smoothening
    img0_smooth = ndi.filters.gaussian_filter(img0, img0_sigma)

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
    img0_seeds_labeled, _ = ndi.label(img0_seeds)

    # Inverted watershed for segmentation
    img0_seg = watershed(~img0_smooth, img0_seeds_labeled)

    # Removal of dilated nuclei
    img0_seg_nuclei = img0_seg * ~convolve2d(img0_mem.astype(int), dilate(img0_dilation).astype(int), mode='same').astype(bool)

    # Removal of background based on colocalization channel 2
    img0_thresh_bg = flt.threshold_triangle(img2)
    img0_smooth_thresh_bg = img2>img0_thresh_bg
    img0_seg_clean = img0_seg_nuclei * img0_smooth_thresh_bg

    # Removal of 'debris'
    img0_seg_labeled, img0_seg_labels = ndi.measurements.label(img0_seg_clean)
    img0_seg_label_size = [(img0_seg_labeled == label).sum() for label in range(img0_seg_labels + 1)]
    for label, size in enumerate(img0_seg_label_size):
        if size < img0_min_size_bg:
            img0_seg_clean[img0_seg_labeled == label] = 0

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
               'img1_2_coloc_scr'         : []}

    # Smoothing for spot detection
    img1_smooth = ndi.filters.gaussian_filter(img1, img1_sigma)
    img2_smooth = ndi.filters.gaussian_filter(img2, img2_sigma)

    for cell_ID in np.unique(img0_seg_clean)[1:]:
        img0_cell_mask = img0_seg_clean==cell_ID

        # Area measurement on cell mask
        results['filename'].append(files[0])
        results['cell_ID'].append(cell_ID)
        results['cell_area'].append(np.count_nonzero(img0_cell_mask))

        # Intensity and spot count measurements
        img1_cell = np.where(np.ma.array(img0_cell_mask, mask=img0_cell_mask==0), img1_smooth, 0)
        results['img1_cell_intensity_mean'].append(np.mean(np.nonzero(img1_cell)))
        results['img1_cell_intensity_std'].append(np.std(np.nonzero(img1_cell)))
        results['img1_spot_count'].append(spot_counter(img1_cell, img1_block_size, img1_thresh, img1_min_distance))

        img2_cell = np.where(np.ma.array(img0_cell_mask, mask=img0_cell_mask==0), img2_smooth, 0)
        results['img2_cell_intensity_mean'].append(np.mean(np.nonzero(img2_cell)))
        results['img2_cell_intensity_std'].append(np.std(np.nonzero(img2_cell)))
        results['img2_spot_count'].append(spot_counter(img2_cell, img2_block_size, img2_thresh, img2_min_distance))

        # Colocalization measurements
        results[['img1_2_coloc_org', 'img1_2_coloc_scr']].append(coloc(img0_cell_mask, img1, img2, scrambles))

        # Export results
        with open(''.join([files[0], '.csv']),'w') as outfile:
            header_string = '\t'.join(results.keys()) + '\n'
            outfile.write(header_string)
            for index in range(len(results['filename'])):
                data_string = '\t'.join([str(results[key][index]) for key in results.keys()]) + '\n'
                outfile.write(data_string)
