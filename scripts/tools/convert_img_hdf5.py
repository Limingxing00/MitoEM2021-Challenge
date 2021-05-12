# -*- coding: utf-8 -*-
# @Time    : 2020/12/10 10:19
# @Author  : Mingxing Li
# @FileName: convert_img_hdf5.py
# @Software: PyCharm
"""
Example of how predicted images can be converted into an H5 file. Notice that,
connected components is applied to label different mitochondria to match instance
segmentation problem.
You should modify the following variables:
    - pred_dir : path to the directory from which the images will be read
    - h5file_name : name of the H5 file to be created (follow the instructions in
                    https://mitoem.grand-challenge.org/Evaluation/ to name the
                    files accordingly)
The H5 file should be saved in the directory where this script was called
"""

import os
import pdb
import h5py
import numpy as np
from skimage.io import imread
from tqdm import tqdm
from skimage import measure, feature
from scipy import ndimage
from PIL import ImageEnhance, Image
from os import path

img_shape = (4096, 4096)
pred_dir = '/braindat/lab/limx/MitoEM2021/MitoEM-H/MitoEM-H/mito_val'
pred_ids = sorted(next(os.walk(pred_dir))[2])
h5file_name = '/braindat/lab/limx/MitoEM2021/MitoEM-H/MitoEM-H/human_val_gt.h5'

# Allocate memory for the predictions
pred_stack = np.zeros((len(pred_ids),) + img_shape, dtype=np.int64)

# Read all the images
for n, id_ in tqdm(enumerate(pred_ids)):
    img = imread(os.path.join(pred_dir, id_))
    pred_stack[n] = img

# Downsample by 2
#pred_stack = pred_stack[:,::2,::2]

# Apply connected components to make instance segmentation
pred_stack = (pred_stack).astype('int64')

# nr_objects = len(np.unique(pred_stack))
# print("Number of objects {}".format(nr_objects-1))

# Create the h5 file (using lzf compression to save space)
h5f = h5py.File(h5file_name, 'w')
# pdb.set_trace()
# pred_stack (100, 4096, 4096)
# pred_stack.max 1014
h5f.create_dataset('dataset_1', data=pred_stack, compression="lzf")
h5f.close()