#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob, os, random, re, sys, math
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from czifile import imread
from matplotlib.patches import Patch
from scipy.signal import convolve2d
from scipy import stats
from skimage import io, feature, filters, segmentation, morphology, measure
from sklearn.cluster import k_means
from ipywidgets import interact, widgets
from IPython.display import set_matplotlib_formats

import interaction_factor as IF
from img2mask import img2mask
from colocalization import sharp, extract_mask, mirror_edges, img_to_blocks, scramble, coloc_image


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_context("notebook")
pd.set_option("display.max_rows", 120)
pd.set_option("display.max_columns", 120)


# # FISH colocalization and areal localization
# <a id='section0'></a>
# ***
# 
# ## Colocalization overview
# 
# 1. [Image import](#section1)
# 2. [DAPI-gated cellular segmentation](#section2)
# 3. [Spot detection](#section3)
# 4. [Colocalization analysis](#section4)
# 5. [Cell-to-cell measurements](#section5)
# 6. [Batch processing](#section6)
# 
# 
# Some of the subsequent code was adapted from the European Molecular Biology Laboratory, Jonas Hartmann, Karin Sasaki, Toby Hodges (© 2018, for cellular segmentation) and from Justin Bois, California Institute of Technology (© 2015, for colocalization in #Functions). Detailed documentation for each function can be found in the accompanying `colocalization.py` file. Everything is distributed under a MIT license.

# ## 1. Image import
# <a id='section1'></a>
# 
# First, the images are imported as numpy array. The filetype (here: .stk) must be specified in order to import the correct images. Once imported, the images are displayed for visual inspection. The channels should be labeled as indicated below.
# 
# - Image 0 – Nuclear label – here: DAPI, 405 nm
# - Image 1 – Colocalization 1 – here: FISH PP7, nm
# - Image 2 – Colocalization 2 – here: FISH MS2, nm
# 
# The subsequent sharp function is used to determine the sharpest slice of the z-stack images. This is done by measuring the average intensity differences between pixel values.

# In[3]:


# File path
number = 2
sharp = False
channel_sharp = 0
img_order = [2, 0, 1]
root = '/Users/beichenberger/Downloads/'
files = glob.glob('{}*{}*.czi'.format(root, number))

for file in files:
    # File shape – (1, 3, 30, 2213, 2752, 1)
    img_import = imread(file)
    
    if sharp:
        # Find sharpest image in _ channel
        img_sharp = sharp(img_import[0,channel_sharp,:,:,:,0])
        # Save sharpest images and reorder
        img_import_stack = [img_import[0,l,img_sharp,:,:,0] for l in range(img_import.shape[1])]
    else:
        # Max project
        img_import_stack = [np.max(img_import[0,l,:,:,:,0], axis=0) for l in range(img_import.shape[1])]
    
    img = np.stack(img_import_stack, axis=0)
    img = [img[i] for i in img_order]

# Check if images are loaded correctly
fig, ax = plt.subplots(1, 3, figsize=(15, 10))
ax[0].set_title('DAPI channel')
ax[0].imshow(img[0], interpolation=None, cmap='gray')
ax[1].set_title('Colocalization channel 1')
ax[1].imshow(img[1], interpolation=None, cmap='gray')
ax[2].set_title('Colocalization channel 2')
ax[2].imshow(img[2], interpolation=None, cmap='gray')
plt.show()


# ## 2. DAPI-gated cellular segmentation
# <a id='section2'></a>
# 
# To do a cellular analysis, all cells are segmented. The parts of this multi-step process are described below.

# ### 2.1 Preprocessing
# To obtain a better segmentation later on, the image acquired through the DAPI channel will be smoothed by a gauss filer. Different sigmas (below) can be used to change 'smoothing intensity'.

# In[ ]:


# Gauss-smoothening
img0_sigma = 5
img0_smooth = ndi.filters.gaussian_filter(img[0], img0_sigma)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title('Raw image')
ax[0].imshow(img[0], interpolation='none', cmap='gray')
ax[1].set_title('Smoothed image')
ax[1].imshow(img0_smooth, interpolation='none', cmap='gray')
plt.show()


# ### 2.2 Thresholding
# A global threshold will be applied to remove background noise and to be left with the DAPI nuclei. A variety of thresholding options can be selected. Currently 'otsu' is used.

# In[ ]:


img0_min_size = 300

# Thresholding and removal of small objects
img0_thresh = filters.threshold_otsu(img0_smooth)
img0_smooth_thresh = img0_smooth>img0_thresh
img0_smooth_thresh_fill = ndi.binary_fill_holes(img0_smooth_thresh).astype(bool)
img0_nuclei = morphology.remove_small_objects(img0_smooth_thresh_fill, img0_min_size)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title('Raw image')
ax[0].imshow(img[0], interpolation='none', cmap='gray')
ax[1].set_title('Thresholded DAPI')
ax[1].imshow(img0_nuclei, interpolation='none', cmap='gray')
plt.show()


# ### 2.3 Connected component labeling
# As some DAPI nuclei currently show up to be combined, a further smoothing and a exact euclidean distance transform is used to separate these combined nuclei. Thereby, undersegmentation artifacts can be minimized.

# In[ ]:


# Labeling, euclidean distance transform, smoothing
img0_dist_trans = ndi.distance_transform_edt(img0_nuclei)
img0_dist_trans_smooth = ndi.filters.gaussian_filter(img0_dist_trans, sigma=img0_sigma)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title('Raw image')
ax[0].imshow(img[0], interpolation='none', cmap='gray')
ax[1].set_title('Smoothed labels')
ax[1].imshow(img0_dist_trans_smooth, interpolation='none', cmap='viridis')
plt.show()


# ### 2.4 Seeding
# Local maxima are used to detect every DAPI nucleus. This provides a seed for the later segmentation.

# In[ ]:


img0_min_distance = 50

# Seeding (and dilating for visualization)
img0_seeds = feature.peak_local_max(img0_dist_trans_smooth, indices=False, min_distance=img0_min_distance)
img0_seeds_labeled, _ = ndi.label(img0_seeds)

vis_size = 20
img0_seeds_labeled_dil = ndi.filters.maximum_filter(img0_seeds_labeled, size=vis_size)
img0_seeds_vis = np.ma.array(img0_seeds_labeled_dil, mask=img0_seeds_labeled_dil==0)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title('Raw image')
ax[0].imshow(img[0], interpolation='none', cmap='gray')
ax[1].set_title('Unique seeds')
ax[1].imshow(img0_dist_trans_smooth, interpolation='none', cmap='viridis')
ax[1].imshow(img0_smooth, interpolation='none', cmap='gray')
ax[1].imshow(img0_seeds_vis, interpolation='none', cmap='Set1')
plt.show()


# ### 2.5 Inverted watershed
# Segmentation using the generated seeds. Inverted watershed or random walker can be used (change name). Cells on the border might slightly skew with the results, but because a 63x objective is required to visualize the FISH spots, removing these cells would result in only very few (if any) cells remaining per image. Furthermore colocalization is analyzed and not total spot count.

# In[ ]:


# Inverted watershed for segmentation
img0_seg = segmentation.watershed(~img0_smooth, img0_seeds_labeled)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title('Raw image')
ax[0].imshow(img[0], interpolation='none', cmap='gray')
ax[1].set_title('Segmented cells')
ax[1].imshow(img0_smooth, interpolation='none', cmap='gray')
ax[1].imshow(np.ma.array(img0_seg, mask=img0_seg==0), interpolation='none', cmap='prism', alpha=0.5)
plt.show()


# ### 2.6 Cytoplasmic extraction
# A threshold based approach in the first colocalization channel to remove background. As the removal on a cellular basis and would mess up labelling, the background is shown as whole.

# In[ ]:


img0_thresh_bg = 500

# Removal of background based on colocalization channel 1
img0_smooth_thresh_bg = img[1]>img0_thresh_bg
img0_seg_clean = img0_seg * img0_smooth_thresh_bg

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title('Raw image')
ax[0].imshow(img[0], interpolation='none', cmap='gray')
ax[1].set_title('Raw thresholding')
ax[1].imshow(img0_smooth, interpolation='none', cmap='gray')
ax[1].imshow(np.ma.array(img0_seg_clean, mask=img0_seg_clean==0), interpolation='none', cmap='prism', alpha=0.5)
plt.show()


# ### 2.7 Debris removal and relabeling
# Removal of small segments which are too small to be 'real' cytoplasm. Furthermore, to have a continuous label, every segmented area is relabeled.

# In[ ]:


img0_min_size_bg = 1_000

# Removal of 'debris'
img0_seg_labeled, img0_seg_labels = ndi.measurements.label(img0_seg_clean)
for label, region in enumerate(measure.regionprops(img0_seg_clean)):
    if region.area < img0_min_size_bg:
        img0_seg_clean[img0_seg_clean == label] = 0
        
# Relabel if 'debris' was removed
new_ID = 1
for curr_ID in np.unique(img0_seg_clean)[1:]:
    if not len(img0_seg_clean[img0_seg_clean==curr_ID]) == 0:
        img0_seg_clean[img0_seg_clean==curr_ID] = new_ID
        new_ID += 1
        
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title('Raw image')
ax[0].imshow(img[0], interpolation='none', cmap='gray')
ax[1].set_title('Debris removal')
ax[1].imshow(img0_smooth, interpolation='none', cmap='gray')
ax[1].imshow(np.ma.array(img0_seg_clean, mask=img0_seg_clean==0), interpolation='none', cmap='prism', alpha=0.5)
plt.show()


# In[ ]:





# ## 3. Spot detection
# <a id='section3'></a>
# 
# To find optimal spot detection parameters in the second image channel, the widgets below can be used. In this pipeline, the spot detection is used filter out false positives. Cells, for example, which don't have any spots but seem to have significant colocalization values.

# ### 3.1 Traditional method

# #### Set intensity thresholds
# Set a threshold to cut off low intensity patches.

# In[ ]:


img1_thresh_bg = 4_000
img2_thresh_bg = 8_000

img1_thresh = img[1]>img1_thresh_bg
img2_thresh = img[2]>img2_thresh_bg

vis_size = 5
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].set_title('C1 – Raw image')
ax[0, 0].imshow(img[1], interpolation='none', cmap='gray')
ax[0, 1].set_title('C1 – Raw spots')
ax[0, 1].imshow(ndi.filters.maximum_filter(img1_thresh, size=vis_size), interpolation='none', cmap='Greens')
ax[1, 0].set_title('C2 – Raw image')
ax[1, 0].imshow(img[2], interpolation='none', cmap='gray')
ax[1, 1].set_title('C2 – Raw spots')
ax[1, 1].imshow(ndi.filters.maximum_filter(img2_thresh, size=vis_size), interpolation='none', cmap='Greens')
plt.show()


# #### Filter spots
# A size and circularity threshold can be set. The alternative would be to use a rolling ball. The current python implementation, however, uses nested for-loops and is very inefficient. If there are any faster implementations, one could consider changing.

# In[ ]:


img1_size = (0, 50)
img1_circ = (0, 10)
img2_size = (0, 50)
img2_circ = (0, 10)

# Calculate circularity
circ = lambda r: (4 * math.pi * r.area) / (r.perimeter * r.perimeter)

# Generate labels
img1_seg, _ = ndi.measurements.label(img1_thresh)
img1_seg = measure.label(img1_thresh, connectivity=1)
img2_seg, _ = ndi.measurements.label(img2_thresh)
img2_seg = measure.label(img2_thresh, connectivity=1)

# Exclude regions outside of thresholds C1
for label, region in enumerate(measure.regionprops(img1_seg)):
    label_size = region.area
    label_circ = circ(region)
    if not (label_size >= img1_size[0] and
        label_size <= img1_size[1] and
        label_circ >= img1_circ[0] and
        label_circ <= img1_circ[1]):
        img1_seg[img1_seg == label] = 0
        
# Exclude regions outside of thresholds C2
for label, region in enumerate(measure.regionprops(img2_seg)):
    label_size = region.area
    label_circ = circ(region)
    if not (label_size >= img2_size[0] and
        label_size <= img2_size[1] and
        label_circ >= img2_circ[0] and
        label_circ <= img2_circ[1]):
        img2_seg[img2_seg == label] = 0
        
# Relabel C1
new_ID = 1
for curr_ID in np.unique(img1_seg)[1:]:
    if not len(img1_seg[img1_seg==curr_ID]) == 0:
        img1_seg[img1_seg==curr_ID] = new_ID
        new_ID += 1
        
# Relabel C2
new_ID = 1
for curr_ID in np.unique(img2_seg)[1:]:
    if not len(img2_seg[img2_seg==curr_ID]) == 0:
        img2_seg[img2_seg==curr_ID] = new_ID
        new_ID += 1

vis_size = 5
img1_seg_vis = ndi.filters.maximum_filter(img1_seg > 0, size=vis_size)
img2_seg_vis = ndi.filters.maximum_filter(img2_seg > 0, size=vis_size)
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].set_title('C1 – Raw image')
ax[0, 0].imshow(img[1], interpolation='none', cmap='gray')
ax[0, 1].set_title('C1 – Filtered spots')
ax[0, 1].imshow(img1_seg_vis, interpolation='none', cmap='Greens')
ax[1, 0].set_title('C2 – Raw image')
ax[1, 0].imshow(img[2], interpolation='none', cmap='gray')
ax[1, 1].set_title('C2 – Filtered spots')
ax[1, 1].imshow(img2_seg_vis, interpolation='none', cmap='Greens')
plt.show()


# #### Interactive view

# In[ ]:


def test_spot_counter(image, thresh_bg, img_size, img_circ):
    # Threshold
    img_thresh = image>thresh_bg
    
    # Calculate circularity
    circ = lambda r: (4 * math.pi * r.area) / (r.perimeter * r.perimeter)
    
    # Generate labels
    img_seg, _ = ndi.measurements.label(img_thresh)
    img_seg = measure.label(img_thresh, connectivity=1)
    
    # Exclude regions outside of thresholds C1
    for label, region in enumerate(measure.regionprops(img_seg)):
        label_size = region.area
        label_circ = circ(region)
        if not (label_size >= img_size[0] and
            label_size <= img_size[1] and
            label_circ >= img_circ[0] and
            label_circ <= img_circ[1]):
            img_seg[img_seg == label] = 0
    
    vis_size = 10
    seeds_dil = ndi.filters.maximum_filter(img_seg, size=vis_size)
    seeds_vis = np.ma.array(seeds_dil, mask=seeds_dil==0)
    seeds_unique = len(np.unique(img_seg)[1:])
    
    # Visualization
    plt.figure(figsize=(10, 10))
    plt.title(f'Detected spots: {seeds_unique}')
    plt.imshow(image, interpolation='none', cmap='gray')
    plt.imshow(seeds_vis, interpolation='none', cmap='prism', alpha=0.5)
    #plt.savefig('spot_counter.pdf')


# In[ ]:


@interact(t_channel = widgets.ToggleButtons(options=['1', '2'], description='Channel: '),
          t_thresh_bg = widgets.IntSlider(min=0, max=20_000, step=50, value=5000, description='Threshold: '),
          t_size = widgets.IntRangeSlider(min=0, max=5_000, step=10, value=[0, 5_000], description='Size: '),
          t_circ = widgets.FloatRangeSlider(min=0, max=1.0, step=0.05, value=[0, 1], description='Circularity: '))
          
def g(t_channel, t_thresh_bg, t_size, t_circ):
    t_img = img[int(t_channel)]
    test_spot_counter(t_img, t_thresh_bg, t_size, t_circ)


# In[ ]:





# In[ ]:





# ### 3.2 Local maxima (pythonic)

# #### Local thresholding and peak detection

# In[ ]:


img1_block_size = 3
img1_min_distance = 5
img1_threshold_abs = 2_000
img2_block_size = 3
img2_min_distance = 5
img2_threshold_abs = 5_000

# Thresholding and peak detection C1
img1_thresh_local = filters.threshold_local(img[1], img1_block_size, offset=0)
img1_seeds = feature.peak_local_max(img1_thresh_local, indices=False, min_distance=img1_min_distance, threshold_abs=img1_threshold_abs)
img1_seg = measure.label(img1_seeds, connectivity=1)

# Thresholding and peak detection C2
img2_thresh_local = filters.threshold_local(img[2], img2_block_size, offset=0)
img2_seeds = feature.peak_local_max(img2_thresh_local, indices=False, min_distance=img2_min_distance, threshold_abs=img2_threshold_abs)
img2_seg = measure.label(img2_seeds, connectivity=1)

vis_size = 5
img1_seg_vis = ndi.filters.maximum_filter(img1_seeds, size=vis_size)
img2_seg_vis = ndi.filters.maximum_filter(img2_seeds, size=vis_size)
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].set_title('C1 – Raw image')
ax[0, 0].imshow(img[1], interpolation='none', cmap='gray')
ax[0, 1].set_title('C1 – Local maximas')
ax[0, 1].imshow(img1_seg_vis, interpolation='none', cmap='Greens')
ax[1, 0].set_title('C2 – Raw image')
ax[1, 0].imshow(img[2], interpolation='none', cmap='gray')
ax[1, 1].set_title('C2 – Local maximas')
ax[1, 1].imshow(img2_seg_vis, interpolation='none', cmap='Greens')
plt.show()


# #### Interactive view

# In[ ]:


def test_spot_counter(image, sigma, block_size, thresh_abs, min_distance):
    # Set thresholds to reduce noise
    test_smooth = ndi.filters.gaussian_filter(image, sigma)
    thresh = filters.threshold_local(test_smooth, block_size, offset=0) 
    
    # Find and label seeds
    seeds = feature.peak_local_max(thresh, indices=False, min_distance=min_distance, threshold_abs=thresh_abs)
    seeds_labeled, _ = ndi.label(seeds)
    
    vis_size = 10
    seeds_dil = ndi.filters.maximum_filter(seeds_labeled, size=vis_size)
    seeds_vis = np.ma.array(seeds_dil, mask=seeds_dil==0)
    seeds_unique = len(np.unique(seeds_labeled)[1:])
    
    # Visualization
    plt.figure(figsize=(10, 10))
    plt.title(f'Detected spots: {seeds_unique}')
    plt.imshow(image, interpolation='none', cmap='gray')
    plt.imshow(seeds_vis, interpolation='none', cmap='prism', alpha=0.5)
    #plt.savefig('spot_counter.pdf')


# In[ ]:


@interact(t_img = widgets.ToggleButtons(options=['1', '2'], description='Channel: '),
         t_sigma = widgets.IntSlider(min=0, max=20, step=1, value=1, description='Sigma: '),
         t_block_size = widgets.IntSlider(min=1, max=21, step=2, value=5, description='Block size: '),
         t_thresh_abs = widgets.IntSlider(min=0, max=15_000, step=100, value=7_000, description='Treshold: '),
         t_min_distance = widgets.IntSlider(min=0, max=30, step=1, value=5, description='Min. dist.: '))
def g(t_img, t_sigma, t_block_size, t_thresh_abs, t_min_distance):
    test_img = img[int(t_img)]
    test_img = img[2]
    test_spot_counter(test_img, t_sigma, t_block_size, t_thresh_abs, t_min_distance)


# ### 3.3 Local maxima (comparison ImageJ vs. numpy)
# 
# Literal translation of the ImageJ find maxima plugin. Credit of the original port goes to Dominic Waithe. Not adapted to use for the rest of the workflow yet.

# In[ ]:


from PIL import Image

def isWithin(x, y, direction, width, height):
    #Depending on where we are and where we are heading, return the appropriate inequality.
    xmax = width - 1
    ymax = height - 1
    if direction==0: return (y>0)
    elif direction==1: return (x<xmax and y>0)
    elif direction==2: return (x<xmax)
    elif direction==3: return (x<xmax and y<ymax)
    elif direction==4: return (y<ymax)
    elif direction==5: return (x>0 and y<ymax)
    elif direction==6: return (x>0)
    elif direction==7: return (x>0 and y>0)
    return False

def find_local_maxima(img_data):    
    globalMin = np.min(img_data)
    height = img_data.shape[0]
    width = img_data.shape[1]
    dir_x = [0, 1, 1, 1, 0, -1, -1, -1]
    dir_y = [-1, -1, 0, 1, 1, 1, 0, -1]
    out = np.zeros(img_data.shape)
    
    #Goes through each pixel
    for y in range(0, height):
        for x in range(0, width):
            #Reads in the img_data
            v = img_data[y, x]
            #If the pixel is local to the minima of the whole image, can't be maxima.
            if v == globalMin:
                continue
            
            #Is a maxima until proven that it is not.
            isMax = True
            isInner = (y!=0 and y!=height-1) and (x!=0 and x!=width-1)
            for d in range(0,8):
                #Scan each pixel in neighbourhood
                if isInner or isWithin(x,y,d,width,height):
                    #Read the pixels in the neighbourhood.
                    vNeighbour = img_data[y+dir_y[d],x + dir_x[d]]
                    if vNeighbour >v:
                        #We have found there is larger pixel in the neighbourhood.
                        #So this cannot be a local maxima.
                        isMax = False
                        break
            if isMax:
                out[y, x] = 1
    return out

def find_local_maxima_np(img_data):
    #This is the numpy/scipy version of the above function (find local maxima).
    #Its a bit faster, and more compact code.
    
    #Filter data with maximum filter to find maximum filter response in each neighbourhood
    max_out = ndi.filters.maximum_filter(img_data,size=3)
    #Find local maxima.
    local_max = np.zeros((img_data.shape))
    local_max[max_out == img_data] = 1
    local_max[img_data == np.min(img_data)] = 0
    return local_max


# In[ ]:


#img_file = 
ntol = 10 #Noise Tolerance.

img_data = np.array(img_file).astype('uint8')

if len(img_data.shape)>2:
    img_data = (np.sum(img_data, 2)/3.0)
if np.max(img_data)>255 or np.min(img_data)<0:
    print('warning: your image should be scaled between 0 and 255 (8-bit).')

#Finds the local maxima using maximum filter.
local_max = find_local_maxima_np(img_data)

#Find local maxima coordinates
ypts, xpts = np.where(local_max==1)
#Find the corresponding intensities
ipts = img_data[ypts,xpts]

#Changes order from max to min.
ind_pts = np.argsort(ipts)[::-1]
ypts = ypts[ind_pts]
xpts = xpts[ind_pts]
ipts = ipts[ind_pts]

#Create our variables and allocate memory for speed.
types = np.array(local_max).astype(np.int8)
pListx = np.zeros((img_data.shape[0]*img_data.shape[1]))
pListy = np.zeros((img_data.shape[0]*img_data.shape[1]))
width = img_data.shape[1]
height = img_data.shape[0]

#This defines the pixel neighbourhood 8-connected neighbourhood [3x3]
dir_x = [0,  1,  1,  1,  0, -1, -1, -1]
dir_y = [-1, -1,  0,  1,  1,  1,  0, -1]

#At each stage we classify our pixels. We use 2n as we can use more than one definition
#together.
MAXIMUM = 1
LISTED = 2
PROCESSED = 4
MAX_AREA = 8
EQUAL = 16
MAX_POINT = 32
ELIMINATED = 64

maxSortingError = 0
time_array =[]
#Now we iterate through each of a local maxima and prune the sub-maximal peaks.
#This extends the neighbourhood and combines and prunes away unwanted maxima using the
#noise tolerance to decide what counts and what doesn't
for y0, x0, v0 in zip(ypts, xpts, ipts):
    if (types[y0,x0]&PROCESSED) !=0:
        #If processed already then skip this pixel, it won't be maxima.
        continue
    sortingError = True
    while sortingError == True:
        #Our initial pixel 
        pListx[0] = x0
        pListy[0] = y0
        types[y0,x0] |= (EQUAL|LISTED) #Listed and Equal
        listlen = 1
        listI = 0
        
        #isEdgeMAxima = (x0==0 or x0 == width-1 or y0 == 0 or y0 == height -1)
        sortingError = False
        maxPossible = True
        xEqual = float(x0)
        yEqual = float(y0)
        nEqual = 1.0
        
        while listI < listlen:
            #We iteratively add points. This loop will keep going until we have
            #exhausted the neighbourhood.
            #Collect the next point to consider
            x = pListx[listI]
            y = pListy[listI]
            #Is our point legal. //not necessary, but faster than isWithin.
            #With subsequent 'OR' statement the first arguement is evaluated
            #and then only the second if the first is false.
            isInner = (y != 0 and y != height -1) and (x!=0 and x != width-1)
            
            for d in range(0,8):
                #Scan the neighbourhood.
                x2 = int(x+dir_x[d])
                y2 = int(y+dir_y[d])
                
                if (isInner or isWithin(x,y,d,width,height)) and (types[y2,x2]&LISTED) ==0:
                    #If the pixel is located legally
                    if types[y2,x2]&PROCESSED !=0:
                        #If the pixel is processed already. It won't be maxima.
                        maxPossible = False
                        break;
                    
                    v2 = img_data[y2,x2] #return pixel from neighbourhood.
                    
                    if v2 > v0 + maxSortingError:
                        #We have reached a higher maximum.
                        maxPossible = False
                        break;
                    
                    elif v2 >= v0 - ntol:
                        #If equal or within we add it on.
                        pListx[listlen] = x2
                        pListy[listlen] = y2
                        listlen = listlen+1
                        #We mark it as listed. Because its in our list :-).
                        types[y2,x2] |= LISTED
                        #We are not excluding edge pixels yet.
                        #if (x2==0 or x2 == width-1 or y2==0 or y2==height-1):
                        #    isEdgeMaximum = True
                            #maxPossible = False
                            #break

                        if v2==v0:
                            #This point is equal to our maxima.
                            types[y2,x2] |= EQUAL
                            #We have to merge the coordinates.
                            xEqual += x2
                            yEqual += y2
                            nEqual += 1
            listI +=1
        #if sortingError:
            #If our point x0, y0 was not true maxima and we reach a bigger one, start again.
            #for listI in range(0,Listlen):
        #   types[pListy[0:listlen],pListx[0:listlen]] =0
        #else:
        if maxPossible == True:
            resetMask = ~(LISTED)
        else:
            resetMask = ~(LISTED|EQUAL)
        #Now we calculate the x and y-coordinates, if there were any equal.
        xEqual /= nEqual
        yEqual /= nEqual
        minDist2 = 1e20
        nearestI = 0
        #This makes sure it has same output as the fiji plugin. Not strictly needed.
        xEqual = round(xEqual)
        yEqual = round(yEqual)
        x = pListx[0:listlen].astype(np.int32)
        y = pListy[0:listlen].astype(np.int32)
        types[y,x] &= resetMask
        types[y,x] |= PROCESSED

        if maxPossible:
            types[y,x] |= MAX_AREA
            #This is where we assign the actual maxima location.
            dv = (types[y,x]&EQUAL) !=0
            dist2 = (xEqual-x[dv]).astype(np.float64)**2+(yEqual-y[dv]).astype(np.float64)**2
            indx = np.arange(0,listlen)
            rd_indx = indx[dv]
            nearestI = rd_indx[np.argmin(dist2)]
            x = int(pListx[nearestI])
            y = int(pListy[nearestI])
            types[y,x] |= MAX_POINT
            
#out = types==61
#ypts,xpts = np.where(out)
#count = np.sum(out)


# In[ ]:





# ### 3.4 Count particles per cell
# If more than the 'pure' nucleus is required, it can be diluted using a diamond kernel (add commented code in loop).

# In[ ]:


#img0_dilation = 5
#img0_kernel = morphology.selem.diamond(img0_dilation)
#img0_nuclei = convolve2d(img0_nuclei.astype(int), img0_kernel.astype(int), mode='same').astype(bool)

columns = ['cell_ID', 'c1_nucleus_spots', 'c1_cyto_spots', 'c2_nucleus_spots', 'c2_cyto_spots']
df = pd.DataFrame(columns=columns)

for cell_ID, region in enumerate(measure.regionprops(img0_seg_clean)):
    
    # Generate region masks
    img0_cell_mask = img0_seg_clean==cell_ID
    img0_nucleus_mask = img0_cell_mask * img0_nuclei
    img0_cyto_mask = img0_cell_mask * ~img0_nuclei
    
    # Count spots / location
    img1_nucleus_spots = len(np.unique(img0_nucleus_mask * img1_seg))
    img1_cyto_spots = len(np.unique(img0_cyto_mask * img1_seg))
    img2_nucleus_spots = len(np.unique(img0_nucleus_mask * img2_seg))
    img2_cyto_spots = len(np.unique(img0_cyto_mask * img2_seg))
    
    df = df.append({'cell_ID' : cell_ID,
                    'c1_nucleus_spots' : img1_nucleus_spots,
                    'c1_cyto_spots' : img1_cyto_spots,
                    'c2_nucleus_spots' : img2_nucleus_spots,
                    'c2_cyto_spots' : img2_cyto_spots},
                   ignore_index=True)

df


# In[ ]:





# ### 3.5 Nearest neighbor
# For each spot in channel 1, returns the distance to the closest spot in channel 2. The following formula is used to determine the distance between to points.
# 
# \begin{align}
# \text{C1: (x,y) – C2: (a,b)} \rightarrow \sqrt{(x-a)^2 + (y-b)^2}
# \end{align}

# In[ ]:


def distance(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


# In[ ]:


# Extract regions from spots
img1_regions = measure.regionprops(img1_seg)
img2_regions = measure.regionprops(img2_seg)

# Calculate minimum distance between spots
img1_distances = [min([distance(reg1.centroid, reg2.centroid) for reg2 in img2_regions]) for reg1 in img1_regions]
img2_distances = [min([distance(reg2.centroid, reg1.centroid) for reg1 in img1_regions]) for reg2 in img2_regions]

# Visualize
sns.set(style='ticks', context='talk', font='sans-serif')
sns.distplot(img1_distances, label='Channel 1', )
sns.distplot(img2_distances, label='Channel 2')
plt.xlim(0,)
plt.yticks([])
plt.xlabel('Minimum distance')
plt.ylabel('')
sns.despine(trim=True, bottom=False, left=True)
plt.legend(frameon=False, loc='best')
plt.show()


# ## 4. Colocalization analysis
# <a id='section4'></a>
# 
# There are two main methods of colocaliztion. Intensity based methods such as the pearson correlation coefficient or meanders measure on a pixel-basis. Secondly, object based methods use predefined / segmented objects to give an estimate of colocalization.
# 
# As follows, the pearson correlation coefficient and a novel interaction factor will be discussed. The interaction factor combines both approaches and gives an estimate of statistical important as well as the percentage of overlap between two channels.

# ### 4.1 Pearsons correlation coefficient
# 
# The pearson correlation coefficient (PCC) can be used to quantify colocalization with a intensity weight. The equation is shown below.
# 
# \begin{align}
# r = \frac{\sum_i\left(I_{1,i} - \bar{I}_1\right)\left(I_{2,i} - \bar{I}_2\right)}{\sqrt{\left(\sum_i \left(I_{1,i} - \bar{I}_1\right)^2\right)\left(\sum_i \left(I_{2,i} - \bar{I}_2\right)^2\right)}}.
# \end{align}
# 
# The image set is tested for statistical relevance i.e. if the colocalization is greater than simple variance. This is done via scrambling small blocks of the image. The blocks' height and width are equal to the point spread function (PSF). The PCC is then calculated on the scrambled image blocks. The PSF can be approximated using the rayleigh criterion, which in case of these images / microscopes is as follows.
# 
# \begin{align}
# R_{\text{Confocal}} = \frac{0.4 \times \lambda}{\text{NA}} = \frac{0.4 \times 640\text{ nm}}{1.45} = 278\text{ nm} \rightarrow \text{PSF} \approx 3\text{ px}
# \end{align}
# 
# Due to the fact that the PSF is close to one pixel, every pixel will be scrambled inside the segmented cellular area. Alternatively – if the PSF were larger – the edges would be mirrored to obtain an image that is a multiple of the PSF. This, however, due to the cellular segmentation (resulting in uneven shapes) is not possible. The widget below can be used to visualize the effect of scrambling on the r-value.
# 
# For this analysis, the segmented cytoplasm will give the 'real' PCC value. A box around the segmented mask will be scrambled and will provide information on whether the current cell has a statistically accurate PCC value.

# In[ ]:


def coloc(img1, img2, mirror=True, psf=3, scrambles=200):
    # Mirror edges
    if mirror:
        img1_mirror = mirror_edges(img1, psf)
        img2_mirror = mirror_edges(img2, psf)
    if not mirror:
        img1_mirror = img1
        img2_mirror = img2

    # Generate blocks of both channels
    img1_blocks = img_to_blocks(img1_mirror, psf)
    img2_blocks = img_to_blocks(img2_mirror, psf)

    # Store blocks of channel 2 as flattened array (not scrambled)
    img2_blocks_flat = np.array(img2_blocks).flatten()

    # Scamblin' and obtain R value
    img1_scr = scramble(img1_blocks, img2_blocks_flat, scrambles)
    
    # Unscrambled R value
    img1_unscr, p = stats.pearsonr(np.array(img1_blocks).ravel(), img2_blocks_flat)
    
    # Probablity
    img1_prob = sum(i > img1_unscr for i in img1_scr) / len(img1_scr)
    
    return img1_scr, img1_prob


# In[ ]:


def coloc_cell(img1, img2):
    r_unscr, _ = stats.pearsonr(img1.flatten(), img2.flatten())
    return r_unscr


# In[ ]:


@interact(t_scrambles = widgets.IntSlider(min=0, max=2000, step=10, value=10, description='Scrambles: '),
          t_cell = widgets.IntSlider(min=1, max=len(np.unique(img0_seg_clean)), value=1, description='Cell: '))

def g(t_scrambles, t_cell):
    t_img1 = extract_mask(img0_seg==t_cell, img[1])
    t_img2 = extract_mask(img0_seg==t_cell, img[2])
    
    t_unscr = coloc_cell(t_img1, t_img2)
    t_scr, t_prob = coloc(t_img1, t_img2, scrambles=t_scrambles)
    
    # Visualization
    _ = sns.distplot(t_scr, bins=int(np.sqrt(t_scrambles)), label='Scrambles')
    plt.plot([t_unscr, t_unscr], plt.gca().get_ylim(), '-', label='Real image')
    
    plt.title(f'Scrambled histogram vs. "real" image – Prob: {t_prob}')
    plt.xlabel('Pearson correlation coefficient')
    plt.xlim(-1, 1)
    plt.yticks([])
    plt.ylabel(None)
    plt.legend(frameon=False)
    sns.despine(trim=True, bottom=False, left=True)
    plt.show()


# In[ ]:





# In[ ]:





# ### 4.2 Interaction factor
# 
# The second possibility of quantifying colocalization is using an interaction factor (IF).
# 
# The interaction factor was first described in 2017 by Bermudez-Hernandez et al. and combined elements from pixel based co-localization approaches (PCC) and object based approaches. The IF takes cluster / spot masks as an input. By randomly simulating the location of these clusters and taking the frequency of overlaps, one can get a density independent measurement. The actual IF is a value between 0 and 1 with higher values signifying a higher interaction factor.
# 
# **Note** – as the input requires a spot mask, the traditional spot detection method or the pythonic local maxima method (with minimum distance = 0) must be used.

# #### Image2Mask
# 
# Image2Mask transforms the input images and ROIs into formats which can be used for the actual interaction factor measurements. The img2mask function outputs a tuple containing the following.
# 1. Merged RGB image (masked)
# 2. ROI mask
# 3. Merged RGB image (original)
# 4. (Optional – cluster measurements)

# In[ ]:


cell = 1 # Non-zero value (background)

img1 = img1_seg * img[1]
img2 = img2_seg * img[2]
roi = img0_seg == cell
ret_images = img2mask(img1, img2, roi)


# #### Interaction Factor
# For visualization purposes, two additional parameters get outputed by `IF.calculate_IF()`. These can be used to plot a graph visualizing the calculation process of the interaction factor.

# In[ ]:


my_IF = IF.interaction_factor(ret_images[0], ret_images[1], ret_images[2])

# Channel 1 as reference
my_IF.ref_color = 0
my_IF.nonref_color = 1
IF1 = my_IF.calculate_IF() # IF, p-value, overlap %
IF1_area = my_IF.orig_area_clusters[0] # total area of clusters ch[i]

# Channel 2 as reference
my_IF.plot_IF_curve = True
my_IF.ref_color = 1
my_IF.nonref_color = 0
IF2 = my_IF.calculate_IF()
IF2_area = my_IF.orig_area_clusters[1]

IF_overlap_count = my_IF.orig_num_ov_clusters
IF_overlap_area = my_IF.orig_area_ov_clusters


# In[ ]:


# Visualization
plt.axhline(y=IF1[2])
plt.axvline(x=IF1[0], color='red')
plt.title(f'IF = {IF1[0]}, %-overlap = {str(100*round(IF1[2],3))}')
plt.plot(IF1[3], IF1[4], '-')
plt.xlabel('Interaction Factor')
plt.ylabel('%-overlap')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()


# ## 5. Cell-to-cell measurements
# <a id='section5'></a>
# 
# The colocalization and spot detection functions will be used to record their respective output on a spot basis. In order to track the settings used for the analysis, all parameters are saved together with the measurements. Prior to the measurements, the background will be substracted as represented above ([DAPI-gated cellular segmentation](#section2)). All values are combined and stored in a .csv file.
# 
# **Note** – the following only describes the general logic. For the final pipeline, functions will siplify the process below.

# In[ ]:


# TODO
# Use regionprop.mean_intensity
# Add IF functionality

columns = ['filename', 'cell_ID', 'cell_count', 'cell_area',
           'channel', 'nuclear', 'x_coordinate', 'y_coordinate', 'spot_closest', 'spot_intensity',
           'IF_value', 'IF_p_value', 'IF_overlap_area', 'IF_overlap_count',
           'IF_area_1', 'IF_area_2', 'IF_overlap_1', 'IF_overlap_2']

df = pd.DataFrame(columns=columns)

# Detect spots
# img1_seg / img2_seg

# Interaction factor
IF_img1 = img1_seg * img[1]
IF_img2 = img2_seg * img[2]

for cell_ID in np.unique(img0_seg_clean)[1:]:
    
    # Define cellular area
    img0_cell_mask = img0_seg_clean==cell_ID
    img0_nucleus_mask = img0_cell_mask * img0_nuclei
    
    # Run colocalization (here IF)
    #ret_images = img2mask(IF_img1, IF_img2, img0_cell_mask)
    #my_IF = IF.interaction_factor(ret_images[0], ret_images[1], ret_images[2])
    #my_IF.ref_color, my_IF.nonref_color = 0, 1
    #IF1 = my_IF.calculate_IF()
    #my_IF.ref_color, my_IF.nonref_color = 1, 0
    #IF2 = my_IF.calculate_IF()

    # Bring in the spots
    img1_regions = measure.regionprops(img1_seg)
    img2_regions = measure.regionprops(img2_seg)
    
    # Iterate through all spots (both channels)
    for channel in [1, 2]:
        if channel == 1:
            img_region = img1_regions
            img_coords = [reg.centroid for reg in img1_regions]
            #img_intensity = [reg.mean_intensity for reg in img1_regions] TODO
            img_intensity = [img[1][int(c[0])][int(c[1])] for c in img_coords]
            img_distances = [min([distance(reg1.centroid, reg2.centroid) for reg2 in img2_regions]) for reg1 in img1_regions]
            img_nuclear = [img0_nucleus_mask[int(c[0])][int(c[1])] for c in img_coords]
        if channel == 2:
            img_region = img2_regions
            img_coords = [reg.centroid for reg in img2_regions]
            img_intensity = [img[2][int(c[0])][int(c[1])] for c in img_coords]
            img_distances = [min([distance(reg2.centroid, reg1.centroid) for reg1 in img1_regions]) for reg2 in img2_regions]
            img_nuclear = [img0_nucleus_mask[int(c[0])][int(c[1])] for c in img_coords]
            
        for index, spot in enumerate(img_region):
            
            df = df.append({'filename' : files[0],
                            'cell_ID' : cell_ID,
                            'cell_count' : len(np.unique(img0_seg_clean)),
                            'cell_area' : np.count_nonzero(img0_cell_mask),

                            'channel' : channel,
                            'nuclear' : img_nuclear[index].astype(int),
                            'x_coordinate' : img_coords[index][0],
                            'y_coordinate' : img_coords[index][1],
                            'spot_closest' : img_distances[index],
                            'spot_intensity' : img_intensity[index],

                            #'IF_value' : IF1[0],
                            #'IF_p_value' : IF1[1],
                            #'IF_overlap_area' : my_IF.orig_area_ov_clusters,
                            #'IF_overlap_count' : my_IF.orig_num_ov_clusters,
                            #'IF_area_1' : my_IF.orig_area_clusters[0],
                            #'IF_area_2' : 'hi',
                            #'IF_overlap_1' : IF1[2],
                            #'IF_overlap_2' : IF2[2],
                           }, ignore_index=True)

#df.to_csv(f'{files[0]}.csv', index=False)


# In[ ]:


df


# ## 6. Batch processing
# <a id='section6'></a>
# 
# Batch processing means running complete pipeline over multiple images. The run_pipeline function is located in the file colocalization.py together with all dependencies and helper functions. Simply run the following code in the terminal.
# 
# ```bash
# python colocalization.py -folder='PATH_TO_FOLDER'
# ```

# In[ ]:




