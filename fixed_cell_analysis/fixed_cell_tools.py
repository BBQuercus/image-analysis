# fixed_cell_tools.py
# !/usr/bin/env python
# coding: utf-8

# TODO – make function (img coords to seg)

import glob, os, random, re, sys, math
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.ndimage as ndi

from sklearn.cluster import k_means
from scipy.signal import convolve2d
from scipy import stats
from skimage import io, feature, filters, segmentation, morphology, measure, draw

from czifile import imread
import interaction_factor as IF

####################
# File import #
####################
def get_files(root, czi=False):
    '''
    Returns list of files (either .stk or .czi) and the number of files.
    Parameters
    '''
    if czi:
        files = glob.glob(f'{root}*.czi')
    else:
        files_nd = [file.split('.')[0] for file in glob.glob(f'{root}*.nd')]
        files = [glob.glob(f'{f}*.stk') for f in files_nd]
    return files

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

def import_images(files, sharp=False, sharp_channel=0, czi=False, order=None):
    '''
    Imports files from list of files 
    Parameters
    ----------
    file:
    sharp:
    sharp_channel:
    czi:
    order:
    Returns
    ----------
    img: nd.array
    '''
    if czi:
        img_import = imread(files)
        # File shape – (1, 3, 30, 2213, 2752, 1)
        if sharp:
            img_sharp = sharp(img_import[0,sharp_channel,:,:,:,0])
            img_import_stack = [img_import[0,l,img_sharp,:,:,0] for l in range(img_import.shape[1])]
        else:
            # Max project
            img_import_stack = [np.max(img_import[0,l,:,:,:,0], axis=0) for l in range(img_import.shape[1])]
    else:
        img_import = [io.imread(files[i]) for i in range(len(files))]
        if sharp:
            img_sharp = sharp(img_import[sharp_channel])
            img_import_stack = [img_import[i][img_sharp] for i in range(len(img_import))]
        else:
            # Max project
            img_import_stack = [np.max(img_import[l], axis=0) for l in range(len(img_import))]
    # Combine all channels
    img = np.stack(img_import_stack, axis=0)
    if order:
        img = [img[i] for i in order]
    return img

####################
# Segmentation #
####################
def relabel(img_seg):
    """Relabels labeled images for a continuous labeling.

    Args:
        img_seg (np.array): Array with discontinuous labeling.

    Returns:
        np.array: The relabeled array.

    """
    new_ID = 1
    for curr_ID in np.unique(img_seg)[1:]:
        if not len(img_seg[img_seg==curr_ID]) == 0:
            img_seg[img_seg==curr_ID] = new_ID
            new_ID += 1
    return img_seg

def segment(img, img_bg, img0_sigma=5, img0_min_size=300, img0_min_distance=50, img0_thresh_bg=500, img0_min_size_bg=10_000, img0_dilation=5):
    """Standard DAPI / nuclear based cellular segmentation.

    Args:
        img (np.array): Image array with nuclear labeling.
        img_bg (np.array): Image to be used for cytoplasmic extraction.
        img0_sigma (int): Size of gaussian smoothing kernel.
        img0_min_size (int): Minimum size of objects (removed below).
        img0_min_distance (int): Minimum distance between two nuclei.
        img0_thresh_bg (int): Thresholding of background in img_bg.
        img0_min_size_bg (int): Minimum size of cytoplasm (removed below).
        img0_dilation (int): Size of dilation kernel (expand cytoplasm).

    Returns:
        img0_seg_clean (np.array): Segmented cells with consecutive labels.
        img0_nuclei (np.array): Corresponding nuclear labels.

    """

    # Gauss-smoothening
    img0_smooth = ndi.filters.gaussian_filter(img, img0_sigma)

    # Thresholding and removal of small objects
    img0_thresh = filters.threshold_otsu(img0_smooth)
    img0_smooth_thresh = img0_smooth>img0_thresh
    img0_smooth_thresh_fill = ndi.binary_fill_holes(img0_smooth_thresh).astype(bool)
    img0_nuclei = morphology.remove_small_objects(img0_smooth_thresh_fill, img0_min_size)

    # Labeling, euclidean distance transform, smoothing
    img0_dist_trans = ndi.distance_transform_edt(img0_nuclei)
    img0_dist_trans_smooth = ndi.filters.gaussian_filter(img0_dist_trans, sigma=img0_sigma)

    # Seeding (and dilating for visualization)
    img0_seeds = feature.peak_local_max(img0_dist_trans_smooth, indices=False, min_distance=img0_min_distance)
    img0_seeds_labeled, _ = ndi.label(img0_seeds)

    # Treshold background (in img_bg)
    img0_smooth_thresh_bg = img_bg>img0_thresh_bg

    # Remove small objects and dilute
    img0_smooth_objects = morphology.remove_small_objects(img0_smooth_thresh_bg, min_size=img0_min_size_bg) 
    img0_kernel = morphology.selem.diamond(img0_dilation)
    img0_dil = convolve2d(img0_smooth_objects.astype(int), img0_kernel.astype(int), mode='same').astype(bool)
    img1_dil = img0_dil * img_bg

    # Inverted watershed for segmentation
    img0_seg = segmentation.watershed(~img1_dil, img0_seeds_labeled)
    
    # Add small objects again (watershed covers full image)
    img0_seg_clean = img0_seg * img0_dil

    return img0_seg_clean, img0_nuclei

####################
# Colocalization #
####################
def extract_mask(img_mask, img_real, psf=3):
    """Returns a sliced image based on a image mask.

    Args:
        img_mask (np.array.bool): Image mask to create a box of.
        img_real (np.array): Original image.
        psf (int): Size of point spread function in pixels.

    Returns:
        img_extracted (np.array): Segmented cells with consecutive labels.
        img0_nuclei (np.array): Corresponding nuclear labels.

    """
    img_position = np.where(img_mask == True)
    xmin, xmax = min(img_position[0]), max(img_position[0])
    ymin, ymax = min(img_position[1]), max(img_position[1])
    xpad = (xmax-xmin) % psf
    ypad = (ymax-ymin) % psf
    img_extracted = img_real[xmin:xmax-xpad, ymin:ymax-ypad] 
    return img_extracted

def mirror_edges(img, psf=3):
    """Given a 2D image, boundaries are padded by mirroring so that
    the dimensions of the image are multiples for the width of the
    point spread function.

    Args:
        img (np.array): Image to mirror edges.
        psf (int): Size of point spread function in pixels.

    Returns:
        np.array: Image with mirrored edges.

    """
    pad_x = psf - (img.shape[0] % psf)
    pad_y = psf - (img.shape[1] % psf)
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left
    pad_top = pad_y // 2
    pad_bot = pad_y - pad_top
    return np.pad(img, ((pad_top, pad_bot), (pad_left, pad_right)), mode='reflect')

def img_to_blocks(img, psf=3):
    """Converts image to list of square subimages called 'blocks'.

    Args:
        img (np.array): Image with width / height divisible by psf.
        psf (int): Size of point spread function in pixels.

    Returns:
        np.array: 'blocks' of original image with size of the psf.

    """
    if (img.shape[0] % psf != 0 or
        img.shape[1] % psf != 0):
        raise ValueError('Reshape input img to be a multiple of the psf.')

    roi = np.ones_like(img)
    roi_test = np.all
    return np.array([img[i:i + psf, j:j + psf]
                    for i in range(0, img.shape[0], psf)
                        for j in range(0, img.shape[1], psf)
                            if roi_test(roi[i:i + psf, j:j + psf])])

def scramble(blocks_1, blocks_2_flat, scrambles=10):
    """Scrambles blocks_1 'scrambles' times and returns the corresponding Pearson r values.

    Args:
        blocks_1 (np.array): Blocks which are scrambled.
        blocks_2_flat (np.array): Reference blocks which are not scrambled.
        scrambles (int): Number of scrambles to be performed.

    Returns:
        r_scr (list): List of all scrambled pearson r values.

    """
    r_scr = np.zeros(scrambles)
    if len(np.array(blocks_1).ravel()) < 2 or len(blocks_2_flat) < 2:
        return 0
    for i in range(scrambles):
        random.shuffle(blocks_1)
        r, _ = stats.pearsonr(np.array(blocks_1).ravel(), blocks_2_flat)
        r_scr[i] = r
    r_scr = [i for i in r_scr if ~np.isnan(i)]
    return r_scr

def coloc_image(img1, img2, psf=3, scrambles=10):
    """Colocalization of two images and comparison with scrambles.

    Args:
        img1 (np.array): Image array of to be scrambled image.
        img2 (np.array): Image array of the 'static' image.
        psf (int): Point spread function of the longer wavelength.
        scrambles (int): Number of scrambles to be performed.

    Returns:
        img1_unscr (int): Pearson r value of original image.
        img1_scr (list): List of all scrambled pearson r values.
        img1_prob (int): Probability of being statistically significant.

    """
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

def coloc_cellbox(img1, img2, roi, psf=3, scrambles=10):
    """Colocalization of an roi within an image to compare with scrambles.
    A box is placed around the roi to perform the scrambling.

    Args:
        img1 (np.array): Image array of to be scrambled image.
        img2 (np.array): Image array of the 'static' image.
        roi (np.array.bool): Mask of the area of interest.
        psf (int): Point spread function of the longer wavelength.
        scrambles (int): Number of scrambles to be performed.

    Returns:
        img1_unscr (int): Pearson r value of original image.
        img1_scr (list): List of all scrambled pearson r values.
        img1_prob (int): Probability of being statistically significant.

    """
    img1_box = extract_mask(roi, img1)
    img2_box = extract_mask(roi, img2)

    img1_blocks = img_to_blocks(img1_box, psf)
    img2_blocks = img_to_blocks(img2_box, psf)
    
    img1_blocks_flat = np.array(img1_blocks).flatten()
    img2_blocks_flat = np.array(img2_blocks).flatten()
    
    img1_unscr, _ = stats.pearsonr(img1_blocks_flat, img2_blocks_flat)
    img1_scr = scramble(img1_blocks, img2_blocks_flat, scrambles)
    img1_prob = sum(i > img1_unscr for i in img1_scr) / len(img1_scr)
    return img1_unscr, img1_scr, img1_prob

def coloc_cellmask(img1, img2, roi, scramble=False, scrambles=10):
    """Colocalization of an roi within an image to compare with scrambles.
    The pearson r value is calculated of the direct roi (no box).
    The scrambling will scramble pixels instead of psf blocks.

    Args:
        img1 (np.array): Image array of to be scrambled image.
        img2 (np.array): Image array of the 'static' image.
        roi (np.array.bool): Mask of the area of interest.
        scramble (bool): Whether scrambles should be performed.
        scrambles (int): Number of scrambles to be performed.

    Returns:
        img1_unscr (int): Pearson r value of original image.
        *img1_scr (list): List of all scrambled pearson r values.
        *img1_prob (int): Probability of being statistically significant.

    """
    img1_seg = img1 * roi
    img2_seg = img2 * roi
    r_unscr, _ = stats.pearsonr(img1_seg.flatten(), img2_seg.flatten())

    # Do not proceed with scrambling if False
    if not scramble:
        return r_unscr

    # Proceed with scrambling if True
    if scramble:
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

def img2mask(img1, img2, roi):
    """Transforms 'typical' input images and a ROI selection into
    a format which is valid as input for interaction factor calculations.

    Args:
        img1 (np.array): Image array of one image.
        img2 (np.array): Image array of the second image.
        roi (np.array.bool): Mask of the area of interest.

    Returns:
        mask_images (np.array): Merged image in RGB of both input images (labeled).
        roi_selection (np.array.bool): Input roi.
        intensity_images (np.array): Merged image in RGB of both input images (original).
        *image_measurements (dict): Measurements of each cluster.

    """
    cluster_measurements = False
    thresh = 'otsu' #or 'kmeans'
    thresh = [thresh, thresh, thresh]
    cutoff_min, cutoff_max = 0, 255
    cluster_min_size, cluster_max_size = 0, 0
    cluster_min_size = [cluster_min_size, cluster_min_size, cluster_min_size]
    cluster_max_size = [cluster_max_size, cluster_max_size, cluster_max_size]

    if img1.shape != img2.shape:
        raise ValueError('Input images are not of equal shape.')

    imgs = (img1, img2, np.zeros(img1.shape))
    rgb_image = np.dstack(imgs)
    split_images = []

    ### ROI generation
    if img1.shape != roi.shape:
        raise ValueError('Roi is not of equal shape.')
    roi_selection = roi.astype(bool) # true outside ROI
    roi_mask = ~np.array(roi_selection) # true inside ROI
    #else:
        #roi_mask = np.zeros_like(rgb_image[:,:,0])
        #roi_mask = roi_mask.astype(dtype='bool')
        #roi_selection = ~np.array(roi_mask)

    ### Image generation
    # Thresholding on each image
    for i in range(3):
        curr_image = rgb_image[:,:,i]
        curr_image[roi_mask] = 0 # set all pixels outside roi to 0

        # Threshold (otsu or kmeans)
        if(curr_image.min() == curr_image.max()): curr_thresh = 0
        elif thresh[i] == 'otsu':
            curr_thresh = filters.threshold_otsu(curr_image[roi_selection == True])
        elif thresh[i] == 'kmeans':
            pixels = np.expand_dims(curr_image[roi_selection == True].flatten(),axis=1)
            intensity_cutoffs = sorted(k_means(pixels, 3)[0])
            curr_thresh = intensity_cutoffs[2][0]

        curr_image_thresh = curr_image > curr_thresh
        split_images.append([curr_image, curr_image_thresh])

    intensity_images = np.dstack((split_images[0][0], split_images[1][0], split_images[2][0]))
    mask_images = np.dstack((split_images[0][1], split_images[1][1], split_images[2][1]))

    if not cluster_measurements:
        return mask_images, roi_selection, intensity_images

    ### Cluster measurements
    # Measurements on each cluster
    image_prop_list = ['area', 'perimeter', 'centroid', 'weighted_centroid', 
                    'major_axis_length', 'minor_axis_length', 'eccentricity', 
                    'orientation', 'min_intensity', 'mean_intensity', 'max_intensity']

    labeling_structure = [[1,1,1],[1,1,1],[1,1,1]]

    # Measurements for image, area and count of clusters
    image_measurements = {}
    image_measurements['cluster_area_total'] = []
    image_measurements['cluster_count'] = []
    image_measurements['cluster_area_avg'] = []
    image_measurements['overlap_area_total'] = []
    image_measurements['overlap_count'] = []

    # For each color
    for i in range(3):
        curr_image = split_images[i]
        
        # Total area of clusters
        image_measurements['cluster_area_total'].append(curr_image[1].sum())
        
        # Total number of clusters
        clusters_labeled, clusters_count = ndi.label(curr_image[1], structure=labeling_structure)
        image_measurements['cluster_count'].append(clusters_count)
        
        # Average area of cluster
        props = measure.regionprops(clusters_labeled, curr_image[0], coordinates='xy')
        area_avg = 0
        for region in props:
            area_avg += region.area 
        if len(props) > 0: area_avg /= float(len(props))
        image_measurements['cluster_area_avg'].append(area_avg)

        # Overlapping area / count
        area_overlaps = []
        count_overlaps = []
        # For each color (other colors)
        for j in range(3):
            if j != i:
                area_and = np.logical_and(curr_image[1], split_images[j][1])
                area_overlaps.append(area_and.sum())
                clusters_labeled, clusters_count = ndi.label(area_and, structure=labeling_structure)
                count_overlaps.append(clusters_count)
        image_measurements['overlap_area_total'].append(area_overlaps)
        image_measurements['overlap_count'].append(count_overlaps)

    return mask_images, roi_selection, intensity_images, image_measurements

def coloc_IF(img1, img2, roi):
    """Calculations of the interaction factor (according to Bermudez-Hernandez 2017).

    Args:
        img1 (np.array): Image array of one image.
        img2 (np.array): Image array of the second image.
        roi (np.array.bool): Mask of the area of interest.

    Returns:
        IF_overlap_count (np.array): Number of cluster overlaps.
        IF_overlap_area (np.array.bool): Area of cluster overlaps.
        IF1 (list): Interaction Factor (img1), p-Value, overlap (%).
        IF1_area (list): Area of img1 clusters overlapping with img2 clusters.
        IF2 (list): Interaction Factor (img2), p-Value, overlap (%).
        IF2_area (list): Area of img2 clusters overlapping with img2 clusters.

    """
    ret_images = img2mask(img1, img2, roi)
    my_IF = IF.interaction_factor(ret_images[0], ret_images[1], ret_images[2])

    # Channel 1 as reference
    my_IF.ref_color = 0
    my_IF.nonref_color = 1
    IF1 = my_IF.calculate_IF() # IF, p-value, overlap %
    IF1_area = my_IF.orig_area_clusters[0] # total area of clusters ch[i]

    # Channel 2 as reference
    my_IF.ref_color = 1
    my_IF.nonref_color = 0
    IF2 = my_IF.calculate_IF()
    IF2_area = my_IF.orig_area_clusters[1]

    IF_overlap_count = my_IF.orig_num_ov_clusters
    IF_overlap_area = my_IF.orig_area_ov_clusters

    return IF_overlap_count, IF_overlap_area, IF1, IF1_area, IF2, IF2_area

####################
# Spot detection #
####################
def distance(point1, point2):
    """Returns distance between two points (x, y coordinates)

    Args:
        point1 (tuple / list): List of x, y coordinates.
        point2 (tuple / list): List of x, y coordinates.

    Returns:
        int: Direct distance between point1 and point 2.

    """
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def nearest_neighbors(img1_regions, img2_regions):
    """Returns every nearest spot in the other image.

    Args:
        img1_regions (regionprop): List of region proposals (skimage.measure).
        img2_regions (regionprop): List of region proposals (skimage.measure).

    Returns:
        list: Closest distance of spots in img1 (to img2).

    """
    if (len(img1_regions) == 0 or
        len(img2_regions) == 0):
        raise ValueError('No regions to calculate neighbors.')

    return [min([distance(reg1.centroid, reg2.centroid) for reg2 in img2_regions]) for reg1 in img1_regions]

def spots_traditional(img, thresh_bg=4_000, size=(0, 10_000), circularity=(0, 10)):
    """Segments spots in the text-book way using intensity, size, and circularity thresholds.

    Args:
        img (np.array): Input image.
        thresh_bg (int): Intensity threshold of background.
        size (tuple): Size thresholds of individual spots.
        circularity (tuple): Circularity thresholds of individual spots.

    Returns:
        img_seg (np.array): Labeled array of all spots.

    """
    # Threshold
    img_thresh = img>thresh_bg

    # Calculate circularity
    circ = lambda r: (4 * math.pi * r.area) / (r.perimeter * r.perimeter)

    # Generate labels
    img_seg = measure.label(img_thresh, connectivity=1)

    # Exclude regions outside of thresholds
    for label, region in enumerate(measure.regionprops(img_seg)):
        label_size = region.area
        label_circ = 0
        if region.perimeter != 0:
            label_circ = circ(region)
        if not (label_size >= size[0] and
            label_size <= size[1] and
            label_circ >= circularity[0] and
            label_circ <= circularity[1]):
            img_seg[img_seg == label] = 0
    
    return img_seg

def spots_local_maxima(img, sigma=1, block_size=5, min_distance=5, threshold_abs=7_000, spot_size=4):
    """Finds the maximum of individual spots based on local maxima (indirectly segmented).

    Args:
        img (np.array): Input image.
        sigma (int): Size of gaussian smoothing kernel.
        block_size (tuple): Size of matrix for local thresholding.
        min_distance (int): Minimum distance between two spots.
        threshold_abs (int): Intensity threshold of background.
        spot_size (int): Size of spot (in labeled image).

    Returns:
        img_seg (np.array): Labeled array of all spots.

    """
    # Thresholding and peak detection
    img_smooth = ndi.filters.gaussian_filter(img, sigma)
    img_thresh_local = filters.threshold_local(img_smooth, block_size, offset=0)
    img_seeds = feature.peak_local_max(img_thresh_local, indices=False, min_distance=min_distance, threshold_abs=threshold_abs)

    # Dilute x, y coordinates to spots (segmentable)
    img_dil = ndi.filters.maximum_filter(img_seeds, size=math.sqrt(spot_size))
    img_seg = measure.label(img_dil, connectivity=1)

    return img_seg

def spots_algorithm(img, algorithm='LoG', max_sigma=5, num_sigma=10, threshold=0.02):
    """Finds the maximum of individual spots using one of the gaussian / hessian
    algorithms (indirectly segmented). The options are:
    - Laplacian of Gaussian (https://en.wikipedia.org/wiki/Marr–Hildreth_algorithm)
    - Difference of Gaussian (https://en.wikipedia.org/wiki/Difference_of_Gaussians)
    - Difference of Hessian (https://en.wikipedia.org/wiki/Hessian_matrix)

    Args:
        img (np.array): Input image.
        algorithm ('LoG', 'DoG', 'DoH'): Selection of algorithm (see above).
        min_sigma (int): The minimum standard deviation for Gaussian kernel.
            Keep this low to detect smaller blobs.
        max_sigma (int): The maximum standard deviation for Gaussian kernel.
            Keep this high to detect larger blobs.
        num_sigma (int): Number of intermediate values of standard deviations (only for 'LoG')
        threshold (float): The absolute lower bound for scale space maxima.
            Local maxima smaller than thresh are ignored. Reduce this to detect blobs with less intensities.
        overlap (float, 0-1): If the area of two blobs overlaps by a fraction greater than threshold, the smaller blob is eliminated.

    Returns:
        img_seg (np.array): Labeled array of all spots.

    """
    # Measurement
    if algorithm == 'LoG':
        img_meas = feature.blob_log(img, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold).astype(int)
    if algorithm == 'DoG':
        img_meas = feature.blob_dog(img, max_sigma=max_sigma, threshold=threshold).astype(int)
    if algorithm == 'DoH':
        img_meas = feature.blob_doh(img, max_sigma=max_sigma, threshold=threshold).astype(int)
    
    # Conversion to mask
    img_mask = np.zeros(img.shape)
    for i in img_meas:
        rr, cc = draw.circle_perimeter(i[0], i[1], radius=i[2], shape=img_mask.shape)
        img_mask[rr, cc] = 1
    img_seg = measure.label(img_mask, connectivity=1)
    
    return img_seg

def spot_centroid(regions):
    """Returns centroids for a list of regionprops.

    Args:
        regions (regionprops): List of region proposals (skimage.measure).

    Returns:
        list: Centroids of regionprops.

    """
    return [r.centroid for r in regions]

def spot_value(img, centroids):
    """Returns the intensity at a given location.

    Args:
        img (np.array): Image to get intensities of.
        centroids (list): List of coordinates (x, y).

    Returns:
        list: Intensities at given locations.

    """
    return [img[int(c[0])][int(c[1])] for c in centroids]
    