#!/bin/bash

export PATH=/opt/conda/bin:$PATH &&\
cd /braindat/lab/limx/MitoEM2021/CODE/HUMAN/rsunet_retrain_297000_v2 &&\   # You can change the root path for your PC
python connectomics/utils/evaluation/iteration_eval.py



