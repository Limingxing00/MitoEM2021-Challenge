import numpy as np
import SimpleITK as sitk
import h5py
import argparse
import pdb

def random_rgb():
    r = np.random.rand()*255
    g = np.random.rand()*255
    b = np.random.rand()*255
    return np.array([r, g, b]).astype(np.uint8)

parser = argparse.ArgumentParser(description='post processing of the output file!!!')
parser.add_argument('--output_path', type=str, help='output file path')
args = parser.parse_args()
# print("\n",args.output_path)

import sys 
sys.path.append(args.output_path) 
from connectomics.utils.processing.process_mito import *


itk_img = sitk.ReadImage('{}/seed_map.nii.gz'.format(args.output_path))
result = sitk.GetArrayFromImage(itk_img)


print("save in ...")
print("{}/test_output.h5".format(args.output_path))
  
# post-processing
bc_w_result = malis_watershed(result)

# d, h, w = bc_w_result.shape
# # pdb.set_trace()
# color_img = np.zeros((3, d, h, w), dtype='uint8')
# idx_list = np.unique(bc_w_result)
# for idx in idx_list:
    # rgb = random_rgb()
    # color_img[:, bc_w_result == idx] = rgb[:, np.newaxis]

out = sitk.GetImageFromArray(bc_w_result)
sitk.WriteImage(out, "{}/test_output.nii.gz".format(args.output_path))

h5f = h5py.File("{}/test_output.h5".format(args.output_path), 'w')
# pdb.set_trace()
# pred_stack (100, 4096, 4096)
h5f.create_dataset('dataset_1', data=bc_w_result, compression="lzf")
h5f.close()


