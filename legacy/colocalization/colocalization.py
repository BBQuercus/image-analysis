#!/usr/bin/env python
# coding: utf-8
'''
Full colocalization analysis pipeline. To run on a folder with .nd and their corresponding .stk files, use as follows:

python colocalization.py -folder='PATH_TO_FOLDER'

This will result in csv files with all colocalization data in the root directory.
'''

import glob, os, random, re, sys
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import scipy.ndimage as ndi

from tqdm import tqdm
from scipy.signal import convolve2d
from scipy import stats
from skimage import io, feature, filters, segmentation, morphology

def sharp(img):
    '''
    Returns index of the sharpest slice in an image array of shape z, x, y.
    Parameters
    ----------
    n (arr): The first parameter.
    Returns
    ----------
    bool: The return value. True for success, False otherwise.
    '''
    sharpness = []
    array = np.asarray(img, dtype=np.int32)
    for i in range(array.shape[0]):
        y, x = np.gradient(array[i])
        norm = np.sqrt(x**2 + y**2)
        sharpness.append(np.average(norm))
    return sharpness.index(max(sharpness))

def segment(img0, img1, img0_sigma=3, img0_min_size=200, img0_min_distance=50, img0_dilation=5, img0_thresh_bg=1_000, img0_min_size_bg=5_000):
    img0_smooth = ndi.filters.gaussian_filter(img0, img0_sigma)
    img0_thresh = filters.threshold_otsu(img0_smooth)
    img0_smooth_thresh = img0_smooth>img0_thresh
    img0_mem = morphology.remove_small_objects(img0_smooth_thresh, img0_min_size)
    img0_dist_trans = ndi.distance_transform_edt(img0_mem)
    img0_dist_trans_smooth = ndi.filters.gaussian_filter(img0_dist_trans, sigma=img0_sigma)
    img0_seeds = feature.peak_local_max(img0_dist_trans_smooth, indices=False, min_distance=img0_min_distance)
    img0_seeds_labeled, _ = ndi.label(img0_seeds)
    img0_seg = segmentation.watershed(~img0_smooth, img0_seeds_labeled)
    img0_kernel = morphology.selem.diamond(img0_dilation)
    img0_seg_nuclei = img0_seg * ~convolve2d(img0_mem.astype(int), img0_kernel.astype(int), mode='same').astype(bool)
    #img0_thresh_bg = filters.threshold_triangle(img1)
    img0_smooth_thresh_bg = img1>img0_thresh_bg
    img0_seg_clean = img0_seg_nuclei * img0_smooth_thresh_bg
    img0_seg_labeled, img0_seg_labels = ndi.measurements.label(img0_seg_clean)
    img0_seg_label_size = [(img0_seg_labeled == label).sum() for label in range(img0_seg_labels + 1)]
    
    for label, size in enumerate(img0_seg_label_size):
        if size < img0_min_size_bg:
            img0_seg_clean[img0_seg_labeled==label] = 0
    
    new_ID = 1
    for curr_ID in np.unique(img0_seg_clean)[1:]:
        if not len(img0_seg_clean[img0_seg_clean==curr_ID]) == 0:
            img0_seg_clean[img0_seg_clean==curr_ID] = new_ID
            new_ID += 1   
    
    return img0_seg_clean

def extract_mask(img_mask, img_real, psf=3):
    '''
    Returns a sliced image based on a image mask.
    Parameters
    ----------
    img_mask: np.array (bool mask)
    img_real: np.array (values)
    psf: int
    
    Returns
    ----------
    img_extracted: np.array (values)
    '''
    img_position = np.where(img_mask == True)
    xmin, xmax = min(img_position[0]), max(img_position[0])
    ymin, ymax = min(img_position[1]), max(img_position[1])
    xpad = (xmax-xmin) % psf
    ypad = (ymax-ymin) % psf
    img_extracted = img_real[xmin:xmax-xpad, ymin:ymax-ypad] 
    return img_extracted

def mirror_edges(img, psf_width):
    '''
    Given a 2D image, boundaries are padded by mirroring so that
    the dimensions of the image are multiples for the width of the
    point spread function.
    '''
    pad_x = psf_width - (img.shape[0] % psf_width)
    pad_y = psf_width - (img.shape[1] % psf_width)
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left
    pad_top = pad_y // 2
    pad_bot = pad_y - pad_top
    return np.pad(img, ((pad_top, pad_bot), (pad_left, pad_right)), mode='reflect')

def img_to_blocks(img, width=3):
    '''
    Converts image to list of square subimages called 'blocks'.
    '''
    roi = np.ones_like(img)
    roi_test = np.all
    return np.array([img[i:i + width, j:j + width]
                    for i in range(0, img.shape[0], width)
                        for j in range(0, img.shape[1], width)
                            if roi_test(roi[i:i + width, j:j + width])])

def scramble(blocks_1, blocks_2_flat, scrambles):
    '''
    Scrambles blocks_1 n_scramble times and returns the Pearson r values.
    '''
    r_scr = np.zeros(scrambles)
    if len(np.array(blocks_1).ravel()) < 2 or len(blocks_2_flat) < 2:
        return 0
    for i in range(scrambles):
        random.shuffle(blocks_1)
        r, _ = stats.pearsonr(np.array(blocks_1).ravel(), blocks_2_flat)
        r_scr[i] = r
    r_scr = [i for i in r_scr if ~np.isnan(i)]
    return r_scr

def coloc_image(img1, img2, psf=3, scrambles=200):
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
    img1_mirror = mirror_edges(img1, psf)
    img2_mirror = mirror_edges(img2, psf)

    img1_blocks = img_to_blocks(img1_mirror, psf)
    img2_blocks = img_to_blocks(img2_mirror, psf)
    
    img1_blocks_flat = np.array(img1_blocks).flatten()
    img2_blocks_flat = np.array(img2_blocks).flatten()
    
    img1_unscr, _ = stats.pearsonr(img1_blocks_flat, img2_blocks_flat)
    img1_scr = scramble(img1_blocks, img2_blocks_flat, scrambles)
    img1_prob = sum(i > img1_unscr for i in img1_scr) / len(img1_scr)
    return img1_unscr, img1_scr, img1_prob

def coloc_cellbox(img1, img2, psf=3, scrambles=200):
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
    img1_blocks = img_to_blocks(img1, psf)
    img2_blocks = img_to_blocks(img2, psf)
    
    img1_blocks_flat = np.array(img1_blocks).flatten()
    img2_blocks_flat = np.array(img2_blocks).flatten()
    
    img1_unscr, _ = stats.pearsonr(img1_blocks_flat, img2_blocks_flat)
    img1_scr = scramble(img1_blocks, img2_blocks_flat, scrambles)
    img1_prob = sum(i > img1_unscr for i in img1_scr) / len(img1_scr)
    return img1_unscr, img1_scr, img1_prob

def coloc_cellmask(img0, img1, img2, scrambles=200):
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
    r_unscr, _ = stats.pearsonr(img1_seg.flatten(), img2_seg.flatten())
    r_scr = []
    for _ in range(scrambles):
        img1_flat = img1.flatten()
        idx, = np.nonzero(img1_flat)
        img1_flat[idx] = img1_flat[np.random.permutation(idx)]
        img1_flat = img1_flat.reshape(img1_seg.shape)
        r, _ = stats.pearsonr(img1_flat.flatten(), img2_seg.flatten())
        r_scr.append(r)
        
    r_prob = sum(i > r_unscr for i in r_scr) / len(r_scr)
    
    return r_unscr, r_scr, r_prob

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
    thresh = filters.threshold_local(img_, block_size, offset=5) 
    seeds = feature.peak_local_max(thresh, indices=False, min_distance=min_distance)
    seeds_labeled, _ = ndi.label(seeds)
    seeds_unique = len(np.unique(seeds_labeled)[1:]) # '0' to not include background
    return seeds_unique

def run_pipeline(files, root):
    '''
    Runs a colocalization and spot detection pipeline.
    Parameters
    ----------
    index: str
        Base name / index for one set of image files.
    root: dir
        Directory where image files are located.
    Returns
    ----------
    root/index.csv: .csv file
        File with all values in the outfolder.
    '''
    # Parameters
    img1_sigma = 1
    img1_block_size = 5
    img1_min_distance = 5
    img1_thresh = 7000
    img2_sigma = 2
    img2_block_size = 5
    img2_min_distance = 0
    img2_thresh = 8000
    lambdas = [405, 561, 640]

    # Read images
    for f in files:
        if str(lambdas[2]) in f:
            img_sharp = sharp(io.imread(f))
    for f in files:
        for i in range(len(lambdas)):
            if str(lambdas[i]) in f:
                globals()['img_'+str(lambdas[i])] = io.imread(f)[img_sharp]
    img = np.stack([globals()['img_'+str(lambdas[i])] for i in range(len(lambdas))], axis=0)

    df = pd.DataFrame()

    img0_seg_clean = segment(img[0], img[1])
    img1_smooth = ndi.filters.gaussian_filter(img[1], img1_sigma)
    img2_smooth = ndi.filters.gaussian_filter(img[2], img2_sigma)
    unscr_image, scr_image, prob_image = coloc_image(img[1], img[2])
    
    for cell_ID in  tqdm(np.unique(img0_seg_clean)[1:], desc='Cell', leave=False):
        img0_cell_mask = img0_seg_clean==cell_ID
        img1_cell = np.where(np.ma.array(img0_cell_mask, mask=img0_cell_mask==0), img1_smooth, 0)
        img2_cell = np.where(np.ma.array(img0_cell_mask, mask=img0_cell_mask==0), img2_smooth, 0)
        
        img1_coloc, img2_coloc = extract_mask(img0_cell_mask, img[1]), extract_mask(img0_cell_mask, img[2])
        unscr_cellbox, scr_cellbox, prob_cellbox = coloc_cellbox(img1_coloc, img2_coloc)
        unscr_cellmask, scr_cellmask, prob_cellmask = coloc_cellmask(img0_cell_mask, img[1], img[2])
        
        df = df.append({'filename' : files[0],
                        'cell_ID' : cell_ID,
                        'cell_area' : np.count_nonzero(img0_cell_mask),
                        'img1_cell_intensity_mean' : np.mean(np.nonzero(img1_cell)),
                        'img1_cell_intensity_std' : np.std(np.nonzero(img1_cell)),
                        'img1_spot_count' : spot_counter(img1_cell, img1_block_size, img1_thresh, img1_min_distance),
                        'img2_cell_intensity_mean' : np.mean(np.nonzero(img2_cell)),
                        'img2_cell_intensity_std' : np.std(np.nonzero(img2_cell)),
                        'img2_spot_count' : spot_counter(img2_cell, img2_block_size, img2_thresh, img2_min_distance),
                        'img1_2_coloc_unscr_image' : unscr_image,
                        'img1_2_coloc_scr_image' : scr_image,
                        'img1_2_coloc_prob_image' : prob_image,
                        'img1_2_coloc_unscr_cellbox' : unscr_cellbox,
                        'img1_2_coloc_scr_cellbox' : scr_cellbox,
                        'img1_2_coloc_prob_cellbox' : prob_cellbox,
                        'img1_2_coloc_unscr_cellmask' : unscr_cellmask,
                        'img1_2_coloc_scr_cellmask' : scr_cellmask,
                        'img1_2_coloc_prob_cellmask' : prob_cellmask},
                    ignore_index=True)

    df.to_csv(f'{files[0]}.csv', index=False)

def main():
    flags = tf.app.flags
    flags.DEFINE_string('folder', '', 'Main image directory with .nd and .stk files.')
    FLAGS = flags.FLAGS
    
    indices = [file.split('.')[0] for file in os.listdir(FLAGS.folder) if file.endswith('.nd')] 

    for index in tqdm(indices, desc='File'):
        files = glob.glob('{}{}*.stk'.format(FLAGS.folder, index))
        run_pipeline(files, FLAGS.folder)

if __name__ == "__main__":
    main()
