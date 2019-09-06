#!/usr/bin/env python
# coding: utf-8

'''
Full colocalization analysis pipeline. To run on a folder with .nd and their corresponding .stk files, use as follows:

python spot_counting.py -folder='PATH_TO_FOLDER'

This will result in csv files with all colocalization data in the root directory.
'''

import glob, os, random, re, sys, math
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from fixed_cell_tools import *


def main():
    flags = tf.app.flags
    flags.DEFINE_string('folder', '', 'Main image directory with .nd and .stk files.')
    FLAGS = flags.FLAGS

    root = FLAGS.folder
    czi=False
    files = get_files(root, czi=czi)

    columns = ['filename', 'cell_ID', 'cell_count', 'cell_area',
            'channel', 'nuclear', 'x_coordinate', 'y_coordinate', 'spot_closest', 'spot_intensity']

    df = pd.DataFrame(columns=columns)

    # File level
    for file in tqdm(files, desc='File'):
        img = import_images(file, sharp=False, sharp_channel=0, czi=czi, order=None)

        # Segment and get spots
        img0_seg, img0_nuclei = segment(img[0], img_bg=img[1])
        img_seg = (spots_local_maxima(img[1], threshold_abs=4_000), 
                   spots_local_maxima(img[2], threshold_abs=7_000))

        # Cellular level
        for cell_ID in tqdm(np.unique(img0_seg), desc='Cell'):
            # Make masks
            img0_cell_mask = img0_seg==cell_ID
            img0_nucleus_mask = img0_cell_mask * img0_nuclei
            img_seg_cell = (img0_cell_mask * img_seg[0], img0_cell_mask * img_seg[1])
            img_regions = (measure.regionprops(img_seg_cell[0]), measure.regionprops(img_seg_cell[1]))

            # Continue if no spots in cell
            if (len(img_regions[0]) == 0 or
                len(img_regions[1]) == 0):
                continue

            img_distance = (nearest_neighbors(img_regions[0], img_regions[1]), nearest_neighbors(img_regions[1], img_regions[0]))
            img_centroid = (spot_centroid(img_regions[0]), spot_centroid(img_regions[1]))
            img_intensity = (spot_value(img[1], img_centroid[0]), spot_value(img[2], img_centroid[1]))
            img_nuclear = (spot_value(img0_nucleus_mask, img_centroid[0]), spot_value(img0_nucleus_mask, img_centroid[1]))

            # Channel level
            for channel in [1, 2]:
                curr_seg = img_seg_cell[channel-1]
                curr_distance = img_distance[channel-1]
                curr_coords = img_centroid[channel-1]
                curr_intensity = img_intensity[channel-1]
                curr_nuclear = img_nuclear[channel-1]

                # Spot level
                for spot_ID in range(len(np.unique(curr_seg)[1:])):
                    df = df.append({'filename' : file,
                                    'cell_ID' : cell_ID,
                                    'cell_count' : len(np.unique(img0_seg)),
                                    'cell_area' : np.count_nonzero(img0_cell_mask),

                                    'spot_ID' : spot_ID,
                                    'channel' : channel,
                                    'nuclear' : curr_nuclear[spot_ID].astype(int),
                                    'x_coordinate' : curr_coords[spot_ID][0],
                                    'y_coordinate' : curr_coords[spot_ID][1],
                                    'spot_closest' : curr_distance[spot_ID],
                                    'spot_intensity' : curr_intensity[spot_ID]
                                }, ignore_index=True)

    df.to_csv(f'{root}outfile.csv', index=False)

if __name__ == "__main__":
    main()
