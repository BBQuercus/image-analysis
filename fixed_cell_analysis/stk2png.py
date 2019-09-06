import glob, os, random, re, sys, math
import numpy as np
import pandas as pd
import tensorflow as tf

from skimage import io


def main():
    flags = tf.app.flags
    flags.DEFINE_string('folder', '', 'Main image directory with .nd and .stk files.')
    flags.DEFINE_string('channel', '', 'Channel wavelength ')
    flags.DEFINE_string('out', '', 'Directory to save files into')
    FLAGS = flags.FLAGS

    root = FLAGS.folder
    channel = FLAGS.channel
    out = FLAGS.out

    files = glob.glob(f'{root}*{channel}*stk')
    
    for i, f in enumerate(files):
        img = io.imread(f)
        img = np.max(img, axis=0)
        io.imsave(f'{out}img-{i}.png', img)
        
if __name__ == "__main__":
    main()