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
import torch
import json
import os
from datetime import datetime



timestamp = datetime.now()
timestamp = timestamp.strftime('%d-%m-%Y_%H%M')

json_fname = f'experiment_{timestamp}.json'
script_name= 'main'


seeds= [888]
base_range = [8, 18]
width_range = [8, 18, 64]


datasets = ['cifar100','cifar100','svhn' ]

job_name = 'job1'

with open(json_fname, 'a') as outfile:
    outfile.write('[')


num_j=0

for dataset in datasets:
    for base_width in base_range:
        for width in width_range:
            if width>=base_width:
                for seed in seeds:

                    # create a param dict for experiment.json
                    job_config = {
                                'seed': seed,
                                'dataset': dataset,
                                'base_width': int(base_width),
                                'width': int(width)
                                }

                    with open(json_fname, 'a') as outfile:
                        outfile.write('\n')
                        json.dump(job_config, outfile, indent=2)
                        outfile.write(',')
                    num_j+=1

                    my_str=f'python -m {script_name} '
                    for k, v in job_config.items():
                        if v==True:
                            my_str+=f'--{k} '
                        elif v==False:
                            pass
                        else:
                            my_str+=f'--{k} {v} '
                    print(my_str)


with open(json_fname, 'rb+') as outfile:
    outfile.seek(-1, os.SEEK_END)
    outfile.truncate()

with open(json_fname, 'a') as outfile:
    outfile.write('\n]')

