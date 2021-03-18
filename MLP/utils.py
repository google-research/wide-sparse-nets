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
import torchvision
from torchvision import datasets, transforms
import numpy as np
import os
from models import MLP1


def pixelwise_normalization(images):
    orig_type= type(images)
    if orig_type==torch.Tensor: images= images.numpy()

    #=== compute the mean and stdev of each pixel across images
    pix_mean= np.mean(images, axis=0)
    pix_stdev= np.std(images, axis=0)

    #=== normalize images pixel-wise
    a= (images - pix_mean)
    images= np.divide(a, pix_stdev, out=np.zeros(a.shape, dtype=float), where=pix_stdev!=0)

    if orig_type==torch.Tensor: images= torch.Tensor(images)

    return images



def get_NWTF(base_width, width_range, fract_freeze_cl, input_size=784, num_classes=10):
    """
    (NWTF for "Number of Weights To Freeze")
    Distributes the number of weights to freeze in models with larger width 
    over the layers fc and cl according to the given parameters. Returns a list of valid combinations (nwtf_cl, nwtf_fc) for each width.

    Parameters
      base_width (int) - the width of the baseline model.
      width_range (list of int) - a list of widths of the larger models.
      fract_freeze_cl (float) - the maximal fraction of weights that may be frozen in the cl layer.
    
    Returns
      NWTF (dict) - width (keys) and list of tuples (nwtf_cl, nwtf_fc) (vals)
    """
    NWTF={width: [(0,0)] for width in width_range} # initialize
    
    # get valid (nwtf_cl, nwtf_fc) tuples for each width
    for width in width_range:
        nwtf_tot=(width-base_width)*(input_size+num_classes) # total number of weights to freeze
        if nwtf_tot>0:
            numw_cl = num_classes*width
            numw_fc = input_size*width
            # setting upper bounds per layer
            nwtf_fc_max = input_size*width-1    # allow all-1 freezes in fc
            nwtf_cl_max = int(num_classes*width*fract_freeze_cl)    # allow fract_freeze_cl freezes in cl
            nwtf_max= nwtf_fc_max+nwtf_cl_max

            if nwtf_tot>nwtf_max: # the tot. num. of weights to freeze exceeds the max allowed
                print(f'Width {width} is too large for this model: nwtf_tot > nwtf_max')
            else:
                M= min(nwtf_tot, nwtf_cl_max)   # setting ntf for cl layer first
                step= int(M/5) if M>0 else 1
                for nwtf_cl in range(0,M+1,step):
                    nwtf_fc= nwtf_tot-nwtf_cl    # the rest to-freeze goes to fc layer
                    if nwtf_fc<=nwtf_fc_max:
                        NWTF[width].append( (nwtf_cl,nwtf_fc) )
    return NWTF



def load_dataset(dataset, dataset_dir, batch_size, test_batch_size, kwargs):
    if dataset=='MNIST':
        train_loader = load_MNIST(data_dir=dataset_dir, split='train', batch_size=batch_size, shuffle=True, kwargs=kwargs)
        test_loader = load_MNIST(data_dir=dataset_dir, split='test', batch_size=test_batch_size, shuffle=False, kwargs=kwargs)
        input_size = 28**2
        num_classes = 10
    else:
        print(f'Error: Dataset {dataset} not available!')

    return train_loader, test_loader, input_size, num_classes



def load_MNIST(data_dir, split, batch_size, shuffle, kwargs):
    """ Load and preprocess MNIST data, return data loader. """
    train_flag = True if split=='train' else False
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(data_dir, train=train_flag, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    
    return data_loader



def evaluate(model, data_loader, normalize_pixelwise, input_size, device, criterion):
    """ Evaluate model, return prediction accuracy and loss. """
    model.eval()
    loss_sum, correct, total=0, 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images= images.reshape(-1, input_size).to(device)
            if normalize_pixelwise: images= pixelwise_normalization(images) 
            batch_size = images.size(0)

            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            loss_sum += len(images)*loss.item()
            correct += (predicted==labels).sum().item()
            total += len(images)

    acc = correct/total
    loss = loss_sum/total

    return acc, loss



def save_checkpoint(state, savename):
    """Save model checkpoint."""
    
    # create (sub)dirs if not yet existent
    subfolder = savename.split('/')[0]
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists(f'checkpoints/{subfolder}'):
        os.mkdir(f'checkpoints/{subfolder}')
    # save
    torch.save(state, f'checkpoints/{savename}.ckpt')




