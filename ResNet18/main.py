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

import time
import datetime
import argparse
import random
import warnings
from random import randint
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from utils import *
from get_data import get_dataset
from recorder import Recorder


parser = argparse.ArgumentParser(description='Training a ResNet18 on CIFAR or SVHN.')

parser.add_argument('--seed', type=int, default=None, help='random seed')
parser.add_argument('--half', default=False, action='store_true', 
                    help='set model parameters to half precision (excluding BN layers)')
parser.add_argument('--workers', type=int, default=2, help='number of workers')

#==== training: optimizer and scheduler
parser.add_argument('--optimizer', type=str, default='sgd', 
                    help='optimizer')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--scheduler', type=str, default='multistep', 
                    choices=['multistep', 'cosine'],
                    help='lr scheduler')
parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate', dest='lr')
parser.add_argument('--num_epochs', type=int, default=300,
                    help='number of epochs to run')
parser.add_argument('--dataset', type=str, default='cifar10', 
                    choices=['cifar10', 'cifar100', 'svhn'], 
                    help='dataset')
parser.add_argument('--data_dir', type=str, default='./data', 
                    help='name of the data directory')
parser.add_argument('--mbs', type=int, default=128, help='mini-batch size')

#==== model: arch and sparsity
parser.add_argument('--model_name', type=str, default='resnet18', 
                    help='name of the model architecture')
parser.add_argument('--base_width', type=int, default=64, 
                    help='baseline model width (number of output channels in layer conv1) (default: 64)')
parser.add_argument('--width', type=int, default=64, 
                    help='current model width (default: 64)')
parser.add_argument('--io_only', default=False, action='store_true', 
                    help='if True, sparsify conv layers along IO dims only')

args = parser.parse_args()



def main():

    start_time = time.time()
    print(f'Script start time: {datetime.datetime.now()}')

    device='cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)

    # name of the directory for output files - should be unique for every experiment
    local_savedir =f'{args.dataset}_{args.model_name}_{args.base_width}_{args.width}'
    if args.io_only: local_savedir+=f'_io_only'
    local_savedir+=f'_mbs_{args.mbs}_lr_{args.lr}_seed_{args.seed}'

    # === recorder for saving and loading output files and checkpoints
    rec = Recorder(out_dir=local_savedir, save_ckpt_freq=10)


    # ========== create datasets and dataloaders ==========
    # =====================================================
    print(f'\n>>> Preparing data: {args.dataset}...')

    train_set= get_dataset(args.dataset, args.data_dir, 'train')
    test_set = get_dataset(args.dataset, args.data_dir, 'test')

    train_loader= torch.utils.data.DataLoader(train_set, batch_size=args.mbs,
                      shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.mbs,
                      shuffle=False, num_workers=args.workers, pin_memory=True)

    args.num_in_channels= test_set.nchannels
    args.num_classes= test_set.num_classes
    # =====================================================


    # ========== set up model ==========
    # ==================================

    sparse= args.width > args.base_width

    model, smask, lnames_sorted, num_wtf = prep_model_and_smask()
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.momentum, args.wd)
    scheduler = get_scheduler(args.scheduler, optimizer, args.num_epochs)

    # === save smask
    torch.save(smask, f'checkpoints/{local_savedir}/smask.pt')

    # === save model
    rec.save_full_checkpoint(model, optimizer, scheduler, args, epoch=0, test_acc=0)
    
    with open(f'checkpoints/{local_savedir}/model_printout.txt', 'w') as f:
        print(model, file=f)



    # ========== TRAINING ==========
    # ==============================
    print('\n>>> Starting training!')
    for epoch in range(args.num_epochs):
        
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, 
                                        sparse, smask, lnames_sorted, num_wtf, device)
        # === validate
        test_loss, test_acc = validate(test_loader, model, criterion, device)
        # === save results (TB and stats file)
        rec.add_epoch_result(train_loss, test_loss, train_acc, test_acc,
                             global_step=epoch, walltime=time.time()-start_time )
        # === save model checkpoint (the actual saving frequency is set in the rc)
        rec.save_full_checkpoint(model, optimizer, scheduler, args, epoch, test_acc)
        
        scheduler.step()


    # ========== save final model checkpoint =============

    rec.save_full_checkpoint(model, optimizer, scheduler, args, 
                                epoch= -1, test_acc= test_acc)

    rec.close()



def set_seed(seed=None):

  if seed is not None:
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.backends.cudnn.deterministic = True
      warnings.warn('You have chosen to seed training. '\
                    'This will turn on the CUDNN deterministic setting, '\
                    'which can slow down your training considerably! '\
                    'You may see unexpected behavior when restarting '\
                    'from checkpoints.')


def train(train_loader, model, criterion, optimizer, epoch, 
            sparse, smask, lnames_sorted, num_wtf, device):
    model.train() # switch to training mode
    train_loss = 0
    correct = 0
    total = 0
 
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.half: inputs = inputs.half()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if sparse:
            for lind, lname in enumerate(lnames_sorted):
                if num_wtf[lind]>0:
                    lname_for_statedict = get_lname_for_statedict(lname)
                    for n,layer in model.named_parameters():
                        if n==lname_for_statedict:

                            mask_tensor(layer.grad, smask[lname], args.io_only)

        optimizer.step()

        train_loss+= loss.item()
        _, predicted = outputs.max(1)
        correct+= predicted.eq(targets).sum().item()
        total+= targets.size(0)

        train_loss_avg = train_loss/(batch_idx+1)
        train_acc_avg = 100.*correct/total
        
    return train_loss_avg, train_acc_avg
    


def validate(test_loader, model, criterion, device):
    model.eval() # switch to evaluation mode
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if args.half: inputs = inputs.half()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    acc = 100.*correct/total
    lss = test_loss/(batch_idx+1)
    return lss, acc




def prep_model_and_smask( ltypes=['Linear', 'Conv2d'] ):
    """ 
    Initializes model, creates sparsity mask ("smask") and apply it to model's layers, 
    adjusts parameter initialization for sparse layers (accounting for the effective fan-in).
    """
  
    sparse= args.width > args.base_width
    
    # total number of weights in the _baseline_ model
    num_W_tot_base= get_num_W_tot(args, args.base_width, ltypes)
  
    # ===== setup model
    model = get_model(args, args.width)
  
    # ===== collect model properties required for sparsification
    tensor_dims = get_tensor_dims(model, ltypes) # dimensions of each layer tensor
    num_W = get_num_W(tensor_dims) # num weights of each layer (prod. of corresp. tensor dims)
    num_W_tot= sum(num_W.values()) # total number of weights in current model
    num_W_to_freeze_tot= int(num_W_tot-num_W_tot_base) # total number of weights to freeze
    ctvt= num_W_tot_base/num_W_tot # connectivity
    lnames_sorted= sorted(num_W, key=num_W.get, reverse=True) # layer names sorted by size
  
    # initialize smask (set None for layers that are not sparsified)
    smask= {lname: None for lname in lnames_sorted}
  
    # a list containing the number of weights to freeze for each layer (sorted from large to small)
    num_wtf= get_nwtf(num_W_to_freeze_tot, num_W, tensor_dims, lnames_sorted, args.io_only)
    num_layers_to_sparsify = sum(num_wtf>0) # total number of layers that are to be sparse
    ll_str = '\n'.join([lnames_sorted[lind] for lind in range(num_layers_to_sparsify)])

    print(f'Current number of weights: {num_W_tot}')
    print(f'Target number of weights:  {num_W_tot_base}')
    print(f'Target connectivity: {ctvt:.5f}')
    print(f'==> Remove {num_W_to_freeze_tot} weights.')
    print(f'Weights will be removed from {num_layers_to_sparsify} layers:\n{ll_str}')
  
  
    # ===== create and apply smask to each layer
    for lind in range(num_layers_to_sparsify):
      
        # ===== layer properties
        lname= lnames_sorted[lind]
        lsize= num_W[lname]
        ldims= tensor_dims[lname]
        lnwtf= num_wtf[lind]
        lname_for_statedict= get_lname_for_statedict(lname) # converting layer name to match layer key in model.state_dict()

        print(f'\n>>> Layer {lname}:')

        # ===== make smask for current layer
        print(f'-- Making smask ({lnwtf} weights to remove) ...')
        smask[lname]= make_smask_for_layer( lsize, lnwtf, ldims, args.io_only)
        num_1_in_smask= torch.sum(smask[lname]) # number of ones in smask (corresp to num weights to freeze) in given layer

        if args.io_only:
            kernel_size= np.prod(ldims[-2:]) if len(ldims)==4 else 1
            assert kernel_size*num_1_in_smask==lntf, f"<!> smask is wrong! {kernel_size*num_1_in_smask} not equal {lntf}"
        else:
            assert num_1_in_smask==lnwtf, f"<!> smask is wrong! {num_1_in_smask} is not {lnwtf}"        
    
        # ===== adjust initialization values in sparse layer
        print(f'-- Adjusting weight initialization ...')
        adjust_layer_init(model, lname, lname_for_statedict, ldims, lnwtf, lsize)
    
        # ===== apply smask to layer
        print(f'-- Applying smask ...')
        with torch.no_grad():
            layer= model.state_dict()[lname_for_statedict]
            mask_tensor(layer, smask[lname], args.io_only)

            num_zero=torch.sum(layer==0)
            assert num_zero>=lnwtf, f"<!> layer {lname} not sparsified properly!"

    return model, smask, lnames_sorted, num_wtf





if __name__ == '__main__':
    main()





