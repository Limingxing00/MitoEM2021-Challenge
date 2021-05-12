# -*- coding: utf-8 -*-
# @Time    : 2020/12/10 10:32
# @Author  : Mingxing Li
# @FileName: evaluate.py
# @Software: PyCharm
# !/usr/bin/env python
# coding: utf-8

"""
This script allows you to obtain gt instance and prediction instance matches for the 3D mAP model evaluation. At the end, you can evaluate the mean average precision of your model based on the IoU metric. To do the evaluation, set evaluate to True (default).
"""

import time
import os, sys
import argparse
import numpy as np
import h5py

from vol3d_eval import VOL3Deval
from vol3d_util import seg_iou3d_sorted, readh5_handle, readh5, unique_chunk

global model_num
global path


##### 1. I/O
def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the mean average precision score (mAP) of 3D segmentation volumes')
    parser.add_argument('-gt', '--gt-seg', type=str, default='~/my_ndarray.h5',
                        help='path to ground truth segmentation result')

    parser.add_argument('-p', '--predict-seg', type=str, default='~/my_ndarray.h5',
                        help='path to predicted instance segmentation result')
    # either input the pre-compute prediction score
    parser.add_argument('-ps', '--predict-score', type=str, default='',
                        help='path to a txt or h5 file containing the confidence score for each prediction')
    parser.add_argument('-th', '--threshold', type=str, default='5e3, 3e4',
                        help='get threshold for volume range [possible to have more than 4 ranges, c.f. cocoapi]')
    parser.add_argument('-thc', '--threshold-crumb', type=int, default=2000,
                        help='throw away the imcomplete small mito in the ground truth for a meaningful evaluation')

    parser.add_argument('-cz', '--chunk-size', type=int, default=250,
                        help='for memory-efficient computation, how many slices to load')

    parser.add_argument('-o', '--output-name', type=str, default='map_output',
                        help='output name prefix')
    parser.add_argument('-dt', '--do-txt', type=int, default=1,
                        help='output txt for iou results')
    parser.add_argument('-de', '--do-eval', type=int, default=1,
                        help='do evaluation')
    parser.add_argument('-sl', '--slices', type=str, default="-1",
                        help="slices to load, example: -sl '50, 350'")

    args = parser.parse_args()

    return args


def load_data(args, slices):
    # load data arguments
    pred_seg = readh5_handle(args.predict_seg)
    gt_seg = readh5_handle(args.gt_seg)
    print(args.predict_seg)
    print(args.gt_seg)
    if slices[1] == -1:
        slices[1] = gt_seg.shape[0]

    # check shape match
    sz_gt = np.array(gt_seg.shape)
    sz_pred = pred_seg.shape
    if np.abs((sz_gt - sz_pred)).max() > 0:
        raise ValueError('Warning: size mismatch. gt: {}, pred: '.format(sz_gt, sz_pred))

    if args.predict_score != '':
        print('\t\t Load prediction score')
        # Nx2: pred_id, pred_sc
        if '.h5' in args.predict_score:
            pred_score = readh5(args.predict_score)
        elif '.txt' in args.predict_score:
            pred_score = np.loadtxt(args.predict_score)
        else:
            raise ValueError('Unknown file format for the prediction score')

        if np.any(pred_score.shape == 2):
            raise ValueError('The prediction score should be a Nx2 array')
        if pred_score.shape[1] != 2:
            pred_score = pred_score.T
    else:  # default
        print('\t\t Assign prediction score')
        # assign same weight
        """
        ui = unique_chunk(pred_seg, slices, chunk_size = args.chunk_size, do_count = False)
        ui = ui[ui>0]
        pred_score = np.ones([len(ui),2],int)
        pred_score[:,0] = ui
        """

        # alternative: sort by size
        ui, uc = unique_chunk(pred_seg, slices, chunk_size=args.chunk_size)
        uc = uc[ui > 0]
        ui = ui[ui > 0]
        pred_score = np.ones([len(ui), 2], int)
        pred_score[:, 0] = ui
        pred_score[:, 1] = uc

    thres = np.fromstring(args.threshold, sep=",")
    areaRng = np.zeros((len(thres) + 2, 2), int)
    areaRng[0, 1] = 1e10
    areaRng[-1, 1] = 1e10
    areaRng[2:, 0] = thres
    areaRng[1:-1, 1] = thres
    return gt_seg, pred_seg, pred_score, areaRng, slices


def main():
    """
    Convert the grount truth segmentation and the corresponding predictions to a coco dataset
    to evaluate this dataset. The 3D volume is comparable to a video-type dataset and will therefore
    be converted as a video instance segmentation
    input:
    output: coco_result_vid.json : This file will be written to your current directory and contains
                                    the metadata about the dataset.
    """
    global model_num
    global path
    ## 1. Load data
    start_time = int(round(time.time() * 1000))
    print('\t1. Load data')
    args = get_args()

    def _return_slices():
        # check if args.slices is well defined and return slices array [slice1, sliceN]
        if args.slices == "-1":
            slices = [0, -1]
        else:  # load specific slices only
            try:
                slices = np.fromstring(args.slices, sep=",", dtype=int)
                # test only 2 boundaries, boundary1<boundary2, and boundaries positive
                if (slices.shape[0] != 2) or \
                        slices[0] > slices[1] or \
                        slices[0] < 0 or slices[1] < 0:
                    raise ValueError("\nspecify a valid slice range, ex: -sl '50, 350'\n")
            except:
                print("\nplease specify a valid slice range, ex: -sl '50, 350'\n")
        return slices

    slices = _return_slices()

    gt_seg, pred_seg, pred_score, areaRng, slices = load_data(args, slices)


    model_num = int(args.predict_seg.split("/")[-1][:6]) # 000000 取出来
    path = ("/").join(args.predict_seg.split("/")[:-1]) # 提取save model 路径下的model

    # print(model_num)
    # print(path)
    ## 2. create complete mapping of ids for gt and pred:
    print('\t2. Compute IoU')
    # result_p (2509, 14)
    # pred_score_sorted (2509, 1)

    result_p, result_fn, pred_score_sorted = seg_iou3d_sorted(pred_seg, gt_seg, pred_score, slices, areaRng,
                                                              args.chunk_size, args.threshold_crumb)
    stop_time = int(round(time.time() * 1000))
    print('\t-RUNTIME:\t{} [sec]\n'.format((stop_time - start_time) / 1000))

    ## 3. Evaluation script for 3D instance segmentation
    if args.output_name == '':
        args.output_name = args.predict_seg[:args.predict_seg.rfind('.')]
    v3dEval = VOL3Deval(result_p, result_fn, pred_score_sorted, model_num, path, output_name=args.output_name)
    if args.do_txt > 0:
        v3dEval.save_match_p()
        # v3dEval.write_csv(epoch=, map75=, path=)
        v3dEval.save_match_fn()
    if args.do_eval > 0:
        print('start evaluation')
        # Evaluation
        v3dEval.params.areaRng = areaRng
        v3dEval.accumulate()
        v3dEval.summarize()


if __name__ == '__main__':
    main()
