import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import tqdm
import cv2
import skimage
import csbdeep.utils
import stardist

def _otsu_single(img):
    '''
    '''
    return img>skimage.filters.threshold_otsu(img)

def otsu_batch(imgs):
    '''
    '''
    return [_otsu_single(img) for img in tqdm.tqdm(imgs, desc='Predicting: ')]

def _unet_single(model, img, bit_depth=16):
    '''
    '''
    def __next_power(x, k=2):
        y, power = 0, 1
        while y < x:
            y = k**power
            power += 1
        return y

    pred_img = img * (1./(2**bit_depth - 1))
    pad_bottom = __next_power(pred_img.shape[0]) - pred_img.shape[0]
    pad_right = __next_power(pred_img.shape[1]) - pred_img.shape[1]
    pred_img = cv2.copyMakeBorder(pred_img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
    pred_img = model.predict(pred_img[None,...,None]).squeeze()
    pred_img = pred_img[:pred_img.shape[0]-pad_bottom, :pred_img.shape[1]-pad_right, :]

    return pred_img

def unet_batch(model, imgs):
    '''Predict all images using a UNet model.
    '''
    return [_unet_single(model, img) for img in tqdm.tqdm(imgs, desc='Predicting: ')]

def _stardist_single(model, img, details=True):
    '''
    '''
    pred_img = csbdeep.utils.normalize(img, 1, 99.8, axis=(0, 1))
    pred_labels, pred_details = model.predict_instances(pred_img)
    if details:
        return pred_labels, pred_details
    return pred_labels

def stardist_batch(model, imgs):
    '''Predict all images using the stardist model.
    '''
    return [_stardist_single(model, img, details=False) for img in tqdm.tqdm(imgs, desc='Predicting: ')]

def starnet_single(model_star, model_unet, img, watershed=True):
    '''Combine stardist instance prediction with UNet segmentation.
    '''
    def __instances_to_centroids(instances):
        centroids = [r.centroid for r in skimage.measure.regionprops(instances)]
        centroids = [tuple(int(round(n)) for n in tup) for tup in centroids]
        img_centroids = np.zeros(instances.shape, dtype=np.int)
        for n, c in enumerate(centroids):
            img_centroids[c[0], c[1]] = n
        return img_centroids

    pred_star = _stardist_single(model_star, img, details=False)
    pred_unet = _unet_single(model_unet, img)

    #TODO – Find optimal prediction, prob. with erosion of borders
    img_centroids = __instances_to_centroids(pred_star)
    img_area = 1 - pred_unet[:,:,0]>0.5
    img_area = skimage.morphology.remove_small_holes(img_area).astype(np.int)

    if not watershed:
        return skimage.segmentation.random_walker(~img_area, img_centroids) * img_area
    return skimage.segmentation.watershed(~img_area, img_centroids, watershed_line=True) * img_area

def starnet_batch(model_star, model_unet, imgs, watershed=True):
    '''
    '''
    return [starnet_single(model_star, model_unet, img, watershed) for img in tqdm.tqdm(imgs, desc='Predicting: ')]