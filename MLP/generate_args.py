# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from datetime import datetime
from utils import *

#===== time stamp for experiment file names
timestamp = datetime.now()
timestamp = timestamp.strftime("%d-%m-%Y_%H%M")

script_name = 'main' # main script to be executed

#================================
#     args for main script      #
#================================

seed= 1 # setting random seed for reproducibility

#=====  MODEL  =====
#
model_type= 'MLP1'
no_bias= True       # don't use biases in layers
make_linear= False  # linear activation function (if False, then ReLU)
no_BN= True      # disable BatchNorm
NTK_style= True  # NTK-style parametrization of the network

base_width= 8
all_widths= [8, 32, 128, 216, 328, 512, 635]

fract_freeze_cl= 0 # allowed fraction of all cl-layer weights that may be frozen
dense_only= False  # consider dense models only, no weight freezing


#=====  TRAINING  =====
#
no_ES= True # disable Early Stopping
train_subset_size= 2048 # train on a subset of the train set

mbs= 256        # mini-batch size
max_epochs= 300 # max number of training epochs

#=====  DATASET  =====
#
dataset= 'MNIST'
normalize_pixelwise= True


#=== for NTK-style nets, the LR value is width-dependent
# loading optimized LR values for each width from file
if NTK_style: bta_avg_and_lr= torch.load('optimized_LR_for_NTK_style_MLP1.pt')


# NWTF (for "Num. Weights To Freeze") is a dictionary with 
# key = width
# val = [(nwtf_cl, nwtf_fc)_1, (nwtf_cl, nwtf_fc)_2, ...]
# i.e., a list of valid combinations of weights to freeze for the respective layer (cl and fc)
if dense_only: 
    NWTF = {base_width: [(0,0)]}
else: 
    NWTF = get_NWTF(base_width, all_widths, fract_freeze_cl)


#=== tags for file names
bias_tag='_no_bias' if no_bias else ''
NTK_tag='_NTK_style' if NTK_style else ''
act_fctn='Linear' if make_linear else 'ReLU'

job_configs=[]
for width, val in NWTF.items():
    for nwtf_cl,nwtf_fc in val:

        cur_base_width=width if nwtf_cl==nwtf_fc else base_width

        # compose name for output dir
        output_dir = f'{dataset}_{model_type}_{NTK_tag}'
        output_dir+= f'_base_{cur_base_width}_width_{width}_{act_fctn}{bias_tag}'
        if train_subset_size>0: output_dir+=f'_train_on_{train_subset_size}_samples'
        if normalize_pixelwise: output_dir+=f'_pixelwise_normalization'

        if NTK_style: # get LR from file
            lrkey=f'{cur_base_width}_{width}'
            lr=bta_avg_and_lr[lrkey]
        else:
            lr= 0.1

        config ={
                    'base_width': int(cur_base_width),
                    'width': int(width),
                    'lr': lr,                              
                    'seed': seed,
                    'nwtf_cl': int(nwtf_cl),
                    'nwtf_fc': int(nwtf_fc),
                    'dataset': dataset,
                    'normalize_pixelwise': normalize_pixelwise,
                    'train_subset_size': train_subset_size,
                    'no_ES': no_ES,
                    'max_epochs': max_epochs,
                    'mbs': mbs,
                    'no_bias': no_bias,
                    'NTK_style': NTK_style,
                    'make_linear': make_linear,
                    'no_BN': no_BN,
                    'output_dir': output_dir
                    }
        job_configs.append(config)


for config in job_configs:
    my_str=f'\npython -m {script_name} '
    for k, v in config.items():
        if isinstance(v, bool):
            if v: my_str+=f'--{k} '
        else:
            my_str+=f'--{k} {v} '
    print(my_str)

