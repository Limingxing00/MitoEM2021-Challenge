# -*- coding: utf-8 -*-
# @Time    : 2020/12/11 22:05
# @Author  : Mingxing Li
# @FileName: iteration_eval.py
# @Software: PyCharm

import subprocess

def cal_infer(root_dir, model_id):

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

    # pdb.set_trace()

    # start_epoch, end_epoch = 5000, 200000
    start_epoch, end_epoch = 297000, 297000
    step_epoch = 2500
    model_id = range(start_epoch, end_epoch+step_epoch, step_epoch)

    # _C.SYSTEM.ROOTDIR = "/braindat/lab/limx/MitoEM2021/CODE/Author/baseline/pytorch_connectomics-master" # root_dir
    # commad 中的路径

    root_dir = "/braindat/lab/limx/MitoEM2021/CODE/HUMAN/rsunet_retrain_297000_v2/"


    # validation 输出h5
    # test 不输出h5
    for i in range(len(model_id)): # 不能有空格
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
                4\
                INFERENCE.INPUT_SIZE\
                [100,256,256]\
                INFERENCE.OUTPUT_SIZE\
                [100,256,256]\
                INFERENCE.STRIDE\
                [1,128,128]\
                INFERENCE.PAD_SIZE\
                [0,128,128]\
                INFERENCE.AUG_NUM\
                0\
                ".format(root_dir, root_dir, root_dir, model_id[i], root_dir)

        out = subprocess.run(command, shell=True)
        print(command, "\n |-------------| \n", out, "\n |-------------| \n")



        command = "/opt/conda/bin/python {}connectomics/utils/evaluation/evaluate.py \
             -gt \
             /braindat/lab/limx/MitoEM2021/MitoEM-H/MitoEM-H/human_val_gt.h5 \
             -p \
             {}outputs/inference_output/{:06d}_out_100_256_256_aug_0_pad_0.h5 \
                 -o {}{:06d}".format(root_dir, root_dir, model_id[i], root_dir, model_id[i])

        out = subprocess.run(command, shell=True)
        print(command, "\n |-------------| \n", out, "\n |-------------| \n")
