#!/bin/bash

export PATH=/opt/conda/bin:$PATH &&\
cd /braindat/lab/limx/MitoEM2021/CODE/Author/residual_unet/28_36_48_64_80_non_local/connectomics/model/block &&\
python setup.py install &&\
cd /braindat/lab/limx/MitoEM2021/CODE/HUMAN/rsunet_retrain &&\
python scripts/main.py --config-file configs/MitoEM/MitoEM-R-BC.yaml --checkpoint /braindat/lab/limx/MitoEM2021/CODE/HUMAN/rsunet_retrain/outputs/dataset_output/checkpoint_162500.pth.tar