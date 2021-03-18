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
import torch.utils.data
import numpy as np
import random
import argparse
import os
import pickle
import importlib
import time
from CustomSummaryWriter import *
from models import MLP1, Lin
from utils import *


if __name__ == '__main__':

    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', default=False, action='store_true',
                        help='disables CUDA')
    parser.add_argument('--no_BN', default=False, action='store_true',
                        help='disables BatchNorm')
    parser.add_argument('--no_ES', default=False, action='store_true',
                        help='disable Early Stopping')
    parser.add_argument('--make_linear', default=False, action='store_true', 
                        help='do not apply activation function')
    parser.add_argument('--NTK_style', default=False, action='store_true', 
                        help='use NTK-style model parametrization')
    parser.add_argument('--max_epochs', type=int, default=1, 
                        help='max number of epochs (default: 1')
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset')
    parser.add_argument('--dataset_dir', type=str, default='./data', help='dataset directory')
    parser.add_argument('--normalize_pixelwise', default=False, action='store_true', 
                        help='do pixelwise data normalization')
    parser.add_argument('--model_type', type=str, default='MLP1', choices=['Lin', 'MLP1'], 
                        help='model type (architecture)')
    parser.add_argument('--init_distrib', type=str, default='uniform', 
                        choices=['uniform', 'normal'], 
                        help='probability distribution for parameter initialization; gets overwritten if NTK-style is chosen')
    parser.add_argument('--no_bias', default=False, action='store_true', 
                        help='no bias in the layers')
    parser.add_argument('--base_width', type=int, default=56, 
                        help='number of units in the hidden layer in the baseline model (default: 56)')
    parser.add_argument('--width', type=int, default=56,
                        help='number of units in the hidden layer in the given (sparse) model (default: 56)')
    parser.add_argument('--nwtf_cl', type=int, default=0,
                        help='number of weights to freeze in cl layer')
    parser.add_argument('--nwtf_fc', type=int, default=0,
                        help='number of weights to freeze in fc layer')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--mbs', type=int, default=100, help='mini-batch size')
    parser.add_argument('--train_subset_size', type=int, default=0,
                        help='number of samples if training on a subset of the original train set')
    parser.add_argument('--seed', type=int, default=888, help='random seed')
    parser.add_argument('--output_dir', default='default_dir', type=str, 
                        help='folder name for saving experiment outputs')

    args= parser.parse_args()
    #print('Experiment arguments summary:')
    #print(vars(args))

    # ==== device configuration
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    seed=args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)


    # ========== training and dataset hyper-params ==========
    # =======================================================

    dataset = args.dataset
    dataset_dir = args.dataset_dir
    normalize_pixelwise= args.normalize_pixelwise
    train_subset_size= args.train_subset_size
    no_ES=args.no_ES
    learning_rate = args.lr
    model_type=args.model_type

    ckpt_every=  25
    max_epochs= args.max_epochs   # cut-off value for train loop


    # ========== load dataset ==========
    # ==================================
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if train_subset_size>0: # training on a subset
        train_batch_size= train_subset_size
    else: # training on original whole train set
        train_batch_size= args.mbs
    test_batch_size = 1000

    train_loader, test_loader, input_size, num_classes =\
        load_dataset(dataset, dataset_dir, train_batch_size, test_batch_size, kwargs)
    
    batch_size= args.mbs

    # ========== model hyper-params ==========
    # ========================================

    do_batch_norm = not args.no_BN
    init_distrib= args.init_distrib
    make_linear= args.make_linear
    add_bias= not args.no_bias
    NTK_style= args.NTK_style
    NTK_tag='_NTK_style' if NTK_style else ''
    
    base_width= args.base_width
    width= args.width

    lkeys= ['fc', 'cl']

    # sparsity
    nwtf_fc=args.nwtf_fc
    nwtf_cl=args.nwtf_cl
    ctvt_total=(base_width/width) if width>0 else 1
    sparse= ctvt_total<1

    output_dir=args.output_dir
    writer= CustomSummaryWriter(output_dir)


    # ========== set up model ==========
    # ==================================
    if model_type=='Lin':
        model = Lin(input_size, num_classes, init_distrib, add_bias).to(device)
    else:
        model = MLP1(input_size, num_classes, width, base_width, init_distrib,
                            nwtf_fc, nwtf_cl,
                            do_batch_norm, make_linear, NTK_style, add_bias).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 
    criterion = nn.CrossEntropyLoss()

    if sparse:
        # ==== get smask from model ====
        smask={}
        for lkey in lkeys:
            smask[lkey]= model._modules['layers'][lkey].weight==0

    # ======== save initial model checkpoint
    start_epoch=0
    state= {'epoch': start_epoch, 'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'args': args}
    save_name= f'{output_dir}_init'
    sshproc = save_checkpoint(state, save_name)


    # ============= train ==============
    # ==================================
    train_loss=1  #init
    epoch=start_epoch+1 #init
    best_test_acc=0 #init
    patience= 20
    test_acc_tracker=list(np.zeros(2*patience)) # keep a list of test acc over past some eps
    model.train()


    if train_subset_size>0:
        images_, labels_ = next(iter(train_loader))
        new_train_set = torch.utils.data.TensorDataset(images_, labels_)
        train_loader = torch.utils.data.DataLoader(new_train_set, batch_size=batch_size, shuffle=True)


    while epoch<max_epochs:

        loss_sum, total, correct = 0, 0, 0
        for i, (images, labels) in enumerate(train_loader):

            images= images.reshape(-1, input_size).to(device)
            if normalize_pixelwise: images= pixelwise_normalization(images) 
            labels = labels.to(device)

            # ==== forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += len(images)*loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted==labels).cpu().sum().item()
            total += len(images)

            # ==== backward and optimize
            optimizer.zero_grad()
            
            loss.backward()
            if sparse: # apply smask to gradients
                for lkey in lkeys:
                    if smask[lkey] is not None: # smask is None if layer is not sparsified
                        layer= model._modules['layers'][lkey]
                        layer.weight.grad[ smask[lkey] ] = 0     
            optimizer.step()
        
        # === epoch completed ===

        train_loss = loss_sum/total
        train_acc = correct/total

        # ======== save model checkpoint
        if epoch%ckpt_every==0:
            state= {'epoch': epoch, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'args': args}
            save_name = f'{output_dir}_epoch_{epoch}'
            sshproc = save_checkpoint(state, save_name)

        # ======== evaluate ========
        test_acc, test_loss = evaluate(model, test_loader, normalize_pixelwise, input_size, device, criterion)
        
        # ======== write to TB and stats file
        # (saves both to tb event files and a separate dict called "stats") every epoch
        writer.add_scalars('acc', {'test': test_acc, 'train': train_acc}, 
                            global_step=epoch, walltime=time.time()-start_time )
        writer.add_scalars('loss', {'test': test_loss, 'train': train_loss}, 
                            global_step=epoch, walltime=time.time()-start_time )
        
        # ======== Early Stopping routine
        if not no_ES:
            test_acc_tracker.append(test_acc)
            _=test_acc_tracker.pop(0)
            prev_avg_acc=np.mean(test_acc_tracker[:patience])
            curr_avg_acc=np.mean(test_acc_tracker[patience:])
            if curr_avg_acc<prev_avg_acc and epoch>(2*patience):
                print(f'>>> Early Stopping: epoch {epoch}')
                print(f'* current avg: {curr_avg_acc}')
                print(f'* previous avg: {prev_avg_acc}')
                print(f'(no improvement over past {patience} epochs)')
                break

        # ==== remember best test acc and save checkpoint
        is_best= test_acc > best_test_acc
        best_test_acc= max(test_acc, best_test_acc)

        if is_best:
            state= {'epoch': epoch, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'args': args}
            save_name= f'{output_dir}_best'
            sshproc= save_checkpoint(state, save_name)

        epoch+=1

    writer.close()  # close current event file
    
    # ========== save final model checkpoint =============
    # ====================================================
    state= {'epoch': epoch, 'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'args': args}
    save_name = f'{output_dir}_final'
    print(f'Saving checkpoint as {save_name}')
    sshproc= save_checkpoint(state, save_name)
    sshproc.wait()



