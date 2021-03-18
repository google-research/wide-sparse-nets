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

import torch
import torch.nn as nn
import numpy as np


class Lin(nn.Module):
    def __init__(self, input_size, num_classes, init_distrib, add_bias=True):
        super(Lin, self).__init__()
        self.outputlayer = nn.Linear(input_size, num_classes, bias=add_bias)
        stddev= 1/np.sqrt(input_size)
        self.reinit_parameters(stddev)

    def forward(self, x):
        x = self.outputlayer(x)
        return x

    def reinit_parameters(self, stddev):
        if self.init_distrib=='normal':
            nn.init.normal_(self.outputlayer.weight, mean=0.0, std=stddev)
            if self.outputlayer.bias is not None:
                nn.init.normal_(self.outputlayer.bias, mean=0.0, std=stddev)
        elif self.init_distrib=='uniform':
            nn.init.uniform_(self.outputlayer.weight, -stddev, stddev)
            if self.outputlayer.bias is not None:
                nn.init.uniform_(self.outputlayer.bias, -stddev, stddev)



class MLP1(nn.Module):
    def __init__(self, input_size, num_classes, width, base_width, init_distrib, 
        num_to_freeze_fc, num_to_freeze_cl,
        do_batch_norm=False, make_linear=False, NTK_style=False, add_bias=True):
        super(MLP1, self).__init__()
        
        self.printout=False

        assert init_distrib=='normal' or init_distrib=='uniform',\
            "specified init_distrib unknown!"
        self.init_distrib = init_distrib
        self.do_batch_norm = do_batch_norm
        self.make_linear = make_linear
        self.NTK_style = NTK_style

        self.lkeys = ['fc', 'cl'] # fc: the hidden layer, cl: classifier (last layer)
        num_in = {'fc': input_size, 'cl': width}
        num_out= {'fc': width,      'cl': num_classes}
        self.ntf = {'fc': num_to_freeze_fc, 'cl': num_to_freeze_cl}
        ctvt={}
        for lkey in self.lkeys:
            ctvt[lkey]=1-self.ntf[lkey]/(num_in[lkey]*num_out[lkey])
            if self.printout: print(f'ctvt[{lkey}]= {ctvt[lkey]}')

        self.prefactor= {'fc': 1.0, 'cl': 1.0} # will be adjusted if NTK-style

        ctvt_total= (base_width / width) if width>0 else 1
        sparse= ctvt_total<1
            

        # ===== LAYERS =====
        self.layers = nn.ModuleDict({
                        'fc': nn.Linear(num_in['fc'], num_out['fc'], bias=add_bias),
                        'cl': nn.Linear(num_in['cl'], num_out['cl'], bias=add_bias)
                        })
        if do_batch_norm:
            self.layers.update({'bn': nn.BatchNorm1d(num_out['fc'], eps=1e-05, momentum=0.1, 
                                                        affine=True, track_running_stats=True)})

        # ===== ACTIVATION FUNCTION =====
        if self.make_linear==False:
            self.activation_funct = nn.ReLU()

        # ===== INITIALIZATION ADJUSTMENT =====
        if self.NTK_style:
            if self.printout: print(f'NTK-style parametrization: adjusting initialization ...')
            for lkey in self.lkeys:
                self.prefactor[lkey]= 1/np.sqrt(num_in[lkey])
                if self.printout: print(f'prefactor[{lkey}]= {self.prefactor[lkey]}')
                stddev= 1/np.sqrt(ctvt[lkey])
                if self.printout: print(f'stddev[{lkey}]= {stddev}')
                self.reinit_parameters(lkey, stddev)
        else:
            for lkey in self.lkeys: 
                stddev= 1/np.sqrt(num_in[lkey] * ctvt[lkey])
                self.reinit_parameters(lkey, stddev)


        # ===== SPARSITY MASK =====
        if sparse:
            self.make_and_apply_smask()


    def forward(self, x):

        lkey='fc'
        out = self.prefactor[lkey]*self.layers[lkey](x)
        if self.make_linear==False: out= self.activation_funct(out)
        if self.do_batch_norm==True: out= self.layers['bn'](out)

        lkey='cl'
        out= self.prefactor[lkey]*self.layers[lkey](out)
        
        return out


    def reinit_parameters(self, lkey, stddev):
        if self.init_distrib=='normal':
            if self.printout: print(f'distribution: {self.init_distrib}')
            if self.printout: print(f'lkey={lkey}, stddev={stddev}')
            nn.init.normal_(self.layers[lkey].weight, mean=0.0, std=stddev)
            if self.layers[lkey].bias is not None:
                nn.init.normal_(self.layers[lkey].bias, mean=0.0, std=stddev)
        
        elif self.init_distrib=='uniform':
            nn.init.uniform_(self.layers[lkey].weight, -stddev, stddev)
            if self.layers[lkey].bias is not None:
                nn.init.uniform_(self.layers[lkey].bias, -stddev, stddev)


    def make_and_apply_smask(self):

        smask={}
        for lkey in self.lkeys:
            dims= self.layers[lkey].weight.shape

            # if torch.cuda.is_available():
            #     smask[lkey]= torch.cuda.FloatTensor(dims).uniform_()
            #     r = torch.topk(smask[lkey].view(-1), self.ntf[lkey])
            #     smask[lkey] = torch.cuda.FloatTensor(dims).fill_(0)
            # else:
            #     smask[lkey]= torch.FloatTensor(dims).uniform_()
            #     r = torch.topk(smask[lkey].view(-1), self.ntf[lkey])
            #     smask[lkey] = torch.FloatTensor(dims).fill_(0)
            # update 08/10/20:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            smask[lkey]= torch.FloatTensor(dims, device=device).uniform_()
            # setting top num_to_freeze values in smask to 1;
            # the corresponding values in the weight tensor will be set to zero and frozen
            r= torch.topk(smask[lkey].view(-1), self.ntf[lkey])
            smask[lkey]= torch.FloatTensor(dims, device=device).fill_(0)

            for i, v in zip(r.indices, r.values):
                index = i.item()
                i_col = index%dims[-1]
                i_row = index//dims[-1]
                smask[lkey][i_row, i_col] = 1

            smask[lkey] = smask[lkey].to(bool)
            s=torch.sum(smask[lkey]).item()
            p=100*s/np.prod(dims)
            print(f'applying smask to layer {lkey} - freezing {s} of {np.prod(dims)} weights ({p:.4f}%)')
            with torch.no_grad(): self.layers[lkey].weight[ smask[lkey] ] = 0


