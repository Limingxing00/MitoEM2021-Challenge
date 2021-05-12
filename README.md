# MitoEM2021-Challenge
Here is the official code of the 1st team "VIDAR" on the [MitoEM Challenge](https://mitoem.grand-challenge.org/evaluation/challenge/leaderboard/).

More detial can refer to the paper, [Advanced Deep Networks for 3D Mitochondria Instance Segmentation](https://arxiv.org/abs/2104.07961) and the [report](https://github.com/Limingxing00/MitoEM2021-Challenge/blob/main/MitoEM2021_Mingxing.pptx) on the MitoEM challenge.

🏆 SOTA for 3D Instance Segmentation on MitoEM. Check out [Papers With Code](https://paperswithcode.com/sota/3d-instance-segmentation-on-mitoem) for the 3D Instance Segmentation on MitoEM and more details.

**We publish the validation results of our model on the rat and the human at** [Google Drive.](https://drive.google.com/file/d/17LvjxGKtZb88PCWMPx8XyFCEtwl7DZUf/view?usp=sharing)  
# Dataset
|          | Validation set | Validation set | Test set       |
|----------|----------------|----------------|----------------|
| MitoEM-R | 400×4096 ×4096 | 100×4096 ×4096 | 500×4096 ×4096 |
| MitoEM-H | 400×4096 ×4096 | 100×4096 ×4096 | 500×4096 ×4096 |

# Framework
![framework](https://github.com/Limingxing00/MitoEM2021-Challenge/blob/main/figure/framework.png)

# Installation
You may refer to the [installation](https://github.com/zudi-lin/pytorch_connectomics#installation).  
The additional packages can be installed as: 
``` python
pip install torchsummary waterz malis
```

# Training stage
Take rat for example, in ```rat/configs/MitoEM/```, you need to modify "im_train_rat.json", "mito_train_rat.json" for training sample and "im_val_rat.json" for validation sample and "im_test_human.json", "im_test_rat.json" for test sample. Note the validation GT file is saved as "h5", you can use the [code](https://github.com/donglaiw/MitoEM-challenge/tree/main/aux) to convert the image slices into an H5 file. Because the challenge calculates the "h5" input for evaluation.  

# Training setting
Take rat for example, you can change "rat/configs/MitoEM/MitoEM-R-BC.yaml". Specificly, you may change the ```INPUT_PATH``` and ```INFERENCE_PATH```.

``` yaml
SYSTEM:
  NUM_GPUS: 2
  NUM_CPUS: 8
MODEL:
  ARCHITECTURE: 'rsunet'
  FILTERS: [28, 36, 48, 64, 80]
  INPUT_SIZE: [36, 320, 320]
  OUTPUT_SIZE: [36, 320, 320]
  IN_PLANES: 1
  OUT_PLANES: 2
  LOSS_OPTION: [['DiceLoss', 'WeightedBCE'], ['DiceLoss', 'WeightedBCE']]
  LOSS_WEIGHT: [[0, 1], [0, 1]]
#  LOSS_OPTION: [['WeightedBCE'], ['WeightedBCE']]
#  LOSS_WEIGHT: [[1], [1]]
  TARGET_OPT: ['0','4-2-1'] # Multi-task learning: binary mask, instance segmentation
  WEIGHT_OPT: [['1'],['1']]
DATASET:
  IMAGE_NAME: 'configs/MitoEM/im_train_rat.json'
  LABEL_NAME: 'configs/MitoEM/mito_train_rat.json'
  INPUT_PATH: '/braindat/lab/limx/MitoEM2021/CODE/HUMAN/rsunet_retrain_297000_v2/' # work container
  # inference: save model
  INFERENCE_PATH: '/braindat/lab/limx/MitoEM2021/CODE/HUMAN/rsunet_retrain_297000_v2/'
  OUTPUT_PATH: 'outputs/dataset_output'
  PAD_SIZE: [0, 0, 0] # [16, 128, 128]  # Mirror padding of big chunk
  DO_CHUNK_TITLE: 1 # json file reading
  DATA_CHUNK_NUM: [8, 2, 2] # [8, 2, 2] # block number of each axis.
  DATA_CHUNK_ITER: 2500 # sample times of per chunk
  LABEL_EROSION: 1
  USE_LABEL_SMOOTH: False
  LABEL_SMOOTH: 0.1

SOLVER:
  LR_SCHEDULER_NAME: "WarmupMultiStepLR"
  BASE_LR: 1e-04
  ITERATION_STEP: 1 # How many iterations return loss once
  ITERATION_SAVE: 1500  # save model
  ITERATION_TOTAL: 300000 # total iteration
  SAMPLES_PER_BATCH: 2 #
INFERENCE:
  INPUT_SIZE: [32, 256, 256]
  OUTPUT_SIZE: [32, 256, 256]
  IMAGE_NAME: 'configs/MitoEM/im_val_rat.json'
  OUTPUT_PATH: 'outputs/inference_output'
  OUTPUT_NAME: 'result.h5'
  PAD_SIZE:  [16, 128, 128]
  AUG_MODE: 'mean'
  AUG_NUM: 0
  STRIDE: [16, 128, 128] # [16, 128, 128]
  SAMPLES_PER_BATCH: 48
```
Run the ```main.py``` to start and the results will be saved in ```rat/outputs/inference_output```
```shell
cd rat &&\
python scripts/main.py --config-file configs/MitoEM/MitoEM-R-BC.yaml 
```
If you have checkpoint file:  
```shell
cd rat &&\
python scripts/main.py --config-file configs/MitoEM/MitoEM-R-BC.yaml --checkpoint xxx
```

# Validation stage
By default, AP-75 is calculated on the validation set every certain epoch during the training process. In our experiment, the validation stage needs almost ```45 GB``` of memory, which is mainly due to post-processing for the seed map of size ```100 × 4096 × 4096```.  

Similarly, you can refer to [eval_iter.sh](https://github.com/Limingxing00/MitoEM2021-Challenge/blob/main/eval_iter.sh) to quickly run inference and evaluation programs.

And you can refer ```rat/connectomics/utils/evaluation/iteration_eval.py``` to call the inference and evaluation.
# Segmentation results

|          | AP-75 (Validation set) | 
|:-:|:-:|
| Res-UNet-R+MT | 0.917 | 
| Res-UNet-H+MT | 0.828 | 

**We publish the validation results of our model on the rat and the human at** [Google Drive.](https://drive.google.com/file/d/17LvjxGKtZb88PCWMPx8XyFCEtwl7DZUf/view?usp=sharing)  

![segmentation results](https://github.com/Limingxing00/MitoEM2021-Challenge/blob/main/figure/seg_results.png)

# Model Zoo
Download the pretrained models [here](https://github.com/Limingxing00/MitoEM2021-Challenge/tree/main/outputs/dataset_output).    
|   Model       | Dataset | 
|:-:|:-:|
| checkpoint_085000.pth.tar | Res-UNet-R | 
| checkpoint_297000.pth.tar | Res-UNet-H | 


Please change ```ARCHITECTURE``` in [config file](https://github.com/Limingxing00/MitoEM2021-Challenge/blob/main/configs/MitoEM/MitoEM-R-BC.yaml).

# Citation
If you find this work or code is helpful in your research, please cite:
>@article{li2021advanced,  
  title={Advanced Deep Networks for 3D Mitochondria Instance Segmentation},  
  author={Li, Mingxing and Chen, Chang and Liu, Xiaoyu and Huang, Wei and Zhang, Yueyi and Xiong, Zhiwei},  
  journal={arXiv preprint arXiv:2104.07961},  
  year={2021}  
}  

# TO DO
- [x] upload code
- [x] upload pre-trained models
- [x] Readme
- [x] upload report on the mitoem-isbi2021-challenge
- [ ] more supplementary materials
