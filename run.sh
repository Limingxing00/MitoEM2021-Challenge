#!/bin/bash

export PATH=/opt/conda/bin:$PATH &&\

cd xxx &&\ # please change it
python scripts/main.py --config-file configs/MitoEM/MitoEM-R-BC.yaml 
