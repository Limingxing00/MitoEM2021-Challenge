# -*- coding: utf-8 -*-
# @Time    : 2020/12/11 22:05
# @Author  : Mingxing Li
# @FileName: iteration_eval.py
# @Software: PyCharm

"""
This script looks very complicated, but in fact most of them are default parameters.
There are some important parameters:
SYSTEM.NUM_GPUS
SYSTEM.NUM_CPUS
INFERENCE.INPUT_SIZE: Although the training size is only 32 × 256 × 256, we have empirically found that D=100 is better.
                      (Thanks to the fully convolutional network, the input size is variable)
INFERENCE.STRIDE
INFERENCE.PAD_SIZE
INFERENCE.AUG_NUM: 0 is faster
"""

import subprocess

def cal_infer(root_dir, model_id):
    """
    If you have enough resources, you can use this function during training. 
    Confirm that this line is open. 
    https://github.com/Limingxing00/MitoEM2021-Challenge/blob/dddb388a4aab004fa577058b53c39266e304fc03/connectomics/engine/trainer.py#L423
    """

    command = "/opt/conda/bin/python {}scripts/main.py --config-file\
                {}configs/MitoEM/MitoEM-R-BC.yaml\
                --inference\
                --do_h5\
                --checkpoint\
                {}outputs/dataset_output/checkpoint_{:06d}.pth.tar\
                --opts\
                SYSTEM.ROOTDIR\
                {}\
                SYSTEM.NUM_GPUS\
                2\
                SYSTEM.NUM_CPUS\
                8\
                DATASET.DATA_CHUNK_NUM\
                [1,1,1]\
                INFERENCE.SAMPLES_PER_BATCH\
                2\
                INFERENCE.INPUT_SIZE\
                [100,256,256]\
                INFERENCE.OUTPUT_SIZE\
                [100,256,256]\
                INFERENCE.STRIDE\
                [1,256,256]\
                INFERENCE.PAD_SIZE\
                [0,256,256]\
                INFERENCE.AUG_NUM\
                0\
            ".format(root_dir, root_dir, root_dir, model_id, root_dir)

    out = subprocess.run(command, shell=True)
    print(command, "\n |-------------| \n", out, "\n |-------------| \n")

    command = "/opt/conda/bin/python {}connectomics/utils/evaluation/evaluate.py \
                 -gt \
                 /braindat/lab/limx/MitoEM2021/MitoEM-H/MitoEM-H/human_val_gt.h5 \
                 -p \
                 {}outputs/inference_output/{:06d}_out_100_256_256_aug_0_pad_0.h5 \
             -o {}{:06d}".format(root_dir, root_dir, model_id, root_dir, model_id)

    out = subprocess.run(command, shell=True)
    print(command, "\n |-------------| \n", out, "\n |-------------| \n")



if __name__=="__main__":
    """
    Please note 
    1. Change the gt file! My gt file is in:
    /braindat/lab/limx/MitoEM2021/MitoEM-H/MitoEM-H/human_val_gt.h5
    2. Change configs/MitoEM/MitoEM-R-BC.yaml:
    'unet_residual_3d' for res-unet-r  and 'rsunet' for res-unet-h
    
    reference protocol for 4 gpus:
    python connectomics/utils/evaluation/iteration_eval.py  \
    --bs 8 \
    --naug 0 
    """
    import argparse
    parser = argparse.ArgumentParser(description="Model Inference.")
    parser.add_argument('--model', type=int, default=297000, help='index number of the model')
    parser.add_argument('--root-path', type=str, default="/braindat/lab/limx/MitoEM2021/CODE/HUMAN/rsunet_retrain_297000_v2_biyelunwen", help='root dir path')
    parser.add_argument('--gt-path', type=str, default="/braindat/lab/limx/MitoEM2021/MitoEM-H/MitoEM-H/human_val_gt.h5", help='root dir path')
    parser.add_argument('--ngpus', type=int, default=4, help='gpu number')
    parser.add_argument('--ncpus', type=int, default=8, help='cpu number')
    parser.add_argument('--bs', type=int, default=8, help='total batch size')
    parser.add_argument('--naug', type=int, default=4, help='test time augmentation, 4 or 16')
    parser.add_argument('--stride', nargs='+', default=[1,128,128], type=int, help='basic stride of the sliding window')
    parser.add_argument('--window-size', nargs='+', default=[100,256,256], type=int, help='basic size of the sliding window')
    
    args = parser.parse_args()
    
    start_epoch, end_epoch = args.model, args.model
    step_epoch = 2500
    model_id = range(start_epoch, end_epoch+step_epoch, step_epoch)

    root_dir = args.root_path+"/"

    # validation stage: output h5
    # test stage: don't output h5
    for i in range(len(model_id)):
        command = ["/opt/conda/bin/python {}scripts/main.py ".format(root_dir),
                "--config-file", "{}configs/MitoEM/MitoEM-R-BC.yaml".format(root_dir),
                "--inference",
                "--do_h5",
                "--checkpoint", "{}outputs/dataset_output/checkpoint_{:06d}.pth.tar".format(root_dir, model_id[i]),
                "--opts",
                "SYSTEM.ROOTDIR", "{}".format(root_dir),
                "SYSTEM.NUM_GPUS", str(args.ngpus),
                "SYSTEM.NUM_CPUS", str(args.ncpus),
                "DATASET.DATA_CHUNK_NUM", "[1,1,1]",
                "INFERENCE.SAMPLES_PER_BATCH", str(args.bs),
                "INFERENCE.INPUT_SIZE", str(args.window_size).replace(" ", ""), # replace function for main.py 
                "INFERENCE.OUTPUT_SIZE", str(args.window_size).replace(" ", ""),
                "INFERENCE.STRIDE", str(args.stride).replace(" ", ""),
                "INFERENCE.PAD_SIZE", "[0,128,128]",
                "INFERENCE.AUG_NUM", str(args.naug)]
        command = " ".join(command)
        out = subprocess.run(command, shell=True)
        print(command, "\n |-------------| \n", out, "\n |-------------| \n")
        

        command = ["/opt/conda/bin/python {}connectomics/utils/evaluation/evaluate.py".format(root_dir),
                 "-gt", args.gt_path,
                 "-p", "{}outputs/inference_output/{:06d}_out_{}_{}_{}_stride_{}_{}_{}_aug_{}_pad_0.h5".format(root_dir, model_id[i], 
                                                                                                 args.window_size[0], args.window_size[1], args.window_size[2],
                                                                                                 args.stride[0], args.stride[1], args.stride[2],
                                                                                                 args.naug),
                 "-o", "{}{:06d}_out_{}_{}_{}_stride_{}_{}_{}_aug_{}_pad_0".format(root_dir, model_id[i], 
                                                                                 args.window_size[0], args.window_size[1], args.window_size[2],
                                                                                 args.stride[0], args.stride[1], args.stride[2],
                                                                                 args.naug)]
        command = " ".join(command)
        out = subprocess.run(command, shell=True)
        print(command, "\n |-------------| \n", out, "\n |-------------| \n")

