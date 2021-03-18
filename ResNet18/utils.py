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

import os
import sys
import time
import math
import random
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler

import models


def get_model(args, width):
	""" Initiates a (ResNet18) model with arch specified by the args and with given width. """
	model_name= args.model_name
	half= args.half
	device='cuda' if torch.cuda.is_available() else 'cpu'
	
	model= models.__dict__[model_name](args.num_in_channels, args.num_classes, num_out_conv1=width)
	model= model.to(device)

	if torch.cuda.is_available():
		ngpus= torch.cuda.device_count()
		print(f'>>> Creating model {model_name} (width {width}) on {ngpus} GPU(s)')
		if ngpus>1:
			model= nn.DataParallel(model).cuda()
			cudnn.benchmark = True
	else:
		print(f'>>> Creating model {model_name} (width {width}) on CPU')

	if half:
		print('Using half precision except in BN layer!')
		model= model.half()
		for module in model.modules():
			if (isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d)):
				module.float()
	return model


def get_optimizer(optimizer_name, parameters, lr, momentum=0, weight_decay=0):
	if optimizer_name == 'sgd':
		return optim.SGD(parameters, lr, momentum=momentum, weight_decay=weight_decay)
	elif optimizer_name == 'nesterov_sgd':
		return optim.SGD(parameters, lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
	elif optimizer_name == 'rmsprop':
		return optim.RMSprop(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
	elif optimizer_name == 'adagrad':
		return optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
	elif optimizer_name == 'adam':
		return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


def get_scheduler(scheduler_name, optimizer, num_epochs, **kwargs):
	if scheduler_name == 'constant':
		return lr_scheduler.StepLR(optimizer, num_epochs, gamma=1, **kwargs)
	elif scheduler_name == 'step':
		return lr_scheduler.StepLR(optimizer, 50, gamma=0.1, **kwargs)
	elif scheduler_name == 'multistep':
		return lr_scheduler.MultiStepLR(optimizer, milestones=[50,120,200], gamma=0.1)
	elif scheduler_name == 'exponential':
		return lr_scheduler.ExponentialLR(optimizer, (1e-3) ** (1 / num_epochs), **kwargs)
	elif scheduler_name == 'cosine':
		return lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=0, last_epoch=-1)


from collections import MutableMapping
def flatten(d, parent_key='', sep='_'):
	items = []
	for k, v in d.items():
		new_key = parent_key + sep + k if parent_key else k
		if isinstance(v, MutableMapping):
			items.extend(flatten(v, new_key, sep=sep).items())
		else:
			items.append((new_key, v))
	return dict(items)


def get_tensor_dims(model, ltypes):
	""" Makes dict tensor_dims containing the dimensions of each layer tensor. """
	tensor_dims= {}
	fill_tensor_dims(model, ltypes, tensor_dims)
	tensor_dims= flatten(tensor_dims)
	return tensor_dims


def fill_tensor_dims(model, ltypes, tensor_dims):
	""" Fills the dict tensor_dims with the dims of each layer of type specified in ltypes 
	in the given model. """   
	for lname, child in model.named_children():
		ltype = child._get_name()

		if ltype in ltypes:
			tensor_dims[lname] = list(child.weight.shape)
		else:
			tensor_dims[lname] = {}
			fill_tensor_dims(child, ltypes, tensor_dims[lname])


def get_num_W(tensor_dims):
	""" Makes dict num_W containing the number of weights for each layer. """
	num_W={}
	for k, v in tensor_dims.items():
		num_W[k]= np.prod(v)
	return num_W


def get_num_W_tot(args, width, ltypes=['Linear', 'Conv2d']):
	"""
	Computes the number of weights in the model specified in args and by the argument width.
	Operates by building this model, getting the tensor dims of each relevant layer (specified in ltypes), and summing over the products of dims of each tensor.
	"""
	print(f'\n==> To count the number of weights in model {args.model_name} (width {width}):')
	_model = get_model(args, width)
	tensor_dims = get_tensor_dims(_model, ltypes)
	print(f'... Done counting; deleting model {args.model_name} (width {width}) <==\n')
	del _model # don't need the model any more
	num_W= get_num_W(tensor_dims)  # dict: key= layer name, value= num weights in layer
	num_W_tot= sum(num_W.values()) # total number of weights in model
	return num_W_tot


import bisect
def find_ge(a, x):
	""" Finds leftmost item greater than or equal to x. """
	i = bisect.bisect_left(a, x)
	if i != len(a):
		#ind = np.where(a==a[i])[0][-1]
		return a[i], i
	raise ValueError



def distribute_remainder(remainder, num_layers_to_sparsify, nwtf, lnames_sorted, tensor_dims):
	""" A sub-function used by get_nwtf(). """
	for l_ind in range(num_layers_to_sparsify):

		lname=lnames_sorted[l_ind]
		kernel_size=np.prod(tensor_dims[lname][-2:]) if len(tensor_dims[lname])==4 else 1

		if remainder>=kernel_size:
			nwtf[l_ind]+=kernel_size
			remainder-=kernel_size
		elif (kernel_size-remainder)<kernel_size/2:
			nwtf[l_ind]+=kernel_size
			remainder-=kernel_size
		else:
			pass
		return remainder


def get_nwtf(nwtf_tot, num_W, tensor_dims, lnames_sorted, io_only):
	""" 
	Distribute the total number of weights to freeze over model layers.
  	Note: In case of io_only=True, it is in general not possible to match nwtf_tot exactly, because weights are removed in packs of kernel_size (for most conv layers, kernel_size=9). Therefore, there is going to be a mismatch, num_W_tot will deviate from num_W_tot_base by up to kernel_size.

    Parameters
      nwtf_tot (int) - total number of weights to freeze.
      num_W (dict) - layer names (keys) and number of weights in layer (vals).
      tensor_dims (dict) - layer names (keys) and the dimensions of layer tensor (vals).
      lnames_sorted (list of str) - layer names, sorted by layer size from large to small.

    Returns
      nwtf (list of int) - number of weights to freeze per layer, order corresponding to lnames_sorted.
  	"""

	num_layers = len(lnames_sorted)
	nwtf = np.zeros(num_layers, dtype=int) # init
	
	# list of num. weights in layer, in sorted order (largest first)
	num_W_sorted_list = [num_W[lname] for lname in lnames_sorted]

	# compute  num. weights differences between layers
	num_W_diffs = np.diff(num_W_sorted_list)
	num_W_diffs = [abs(d) for d in num_W_diffs]

	# auxiliary vector for the following dot product to compute the bins
	aux_vect = np.arange( 1,len(num_W_diffs)+1 )

	# the bins of the staggered sparsification: array of max. num. of weights that can be frozen within the given layer before the next-smaller layer gets involved into sparsification
	ntf_lims = [np.dot(aux_vect[:k], num_W_diffs[:k]) for k in range(1,num_layers)]

	# find in which bin nwtf_tot falls - this gives you the number of layers to sparsify
	lim_val, lim_ind = find_ge(ntf_lims, nwtf_tot)
	num_layers_to_sparsify = lim_ind+1

	# base fill: chunks of num. weights that are frozen in each involved layer until all involved layers have equal num. weights remaining
	base_fill = [sum(num_W_diffs[lind:lim_ind]) for lind in range(lim_ind)]
	base_fill.append(0)

	# the rest that is distributed evenly over all involved layers
	rest_tot = nwtf_tot-sum(base_fill)
	rest = int(np.floor(rest_tot/num_layers_to_sparsify))
	nwtf[:num_layers_to_sparsify] = np.array(base_fill)+rest

	# first layer gets the few additional frozen weights when rest_tot is not evenly divisible 
	# by num_layers_to_sparsify
	if io_only:
		ntf_sum=0
		for l_ind in range(num_layers_to_sparsify):
			lname=lnames_sorted[l_ind]
			kernel_size=np.prod(tensor_dims[lname][-2:]) if len(tensor_dims[lname])==4 else 1
			y=int(np.around(nwtf[l_ind]/kernel_size))
			nwtf[l_ind]=y*kernel_size
			ntf_sum+=(y*kernel_size)

		remainder = nwtf_tot - ntf_sum
		remainder = distribute_remainder(remainder, num_layers_to_sparsify, nwtf, lnames_sorted, tensor_dims)
		assert sum(nwtf)+remainder==nwtf_tot, f"<!> nwtf+remainder ({sum(nwtf)+remainder}) not as expected ({nwtf_tot}) ! "
	else:
		rest_mismatch = rest_tot - rest*num_layers_to_sparsify
		nwtf[0]+= rest_mismatch 
		assert sum(nwtf)==nwtf_tot, f"<!> nwtf ({sum(nwtf)}) not as expected ({nwtf_tot}) ! "

	return nwtf



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_lname_for_statedict(lname):
	""" 
	Convert layer name into the form used to address the weight tensor of the module through model_statedict(). 
	"""
	lname_bits= lname.split('_')
	lname_bits.append('weight')
	lname_for_statedict= '.'.join(lname_bits)
	return lname_for_statedict


def get_fanin(lkey, dims, connectivity):
	""" 
	Compute fan-in and "bound" for parameter initialization 
	for layer of type fc (fully-connected) orconv (convolutional). 
	"""
	if 'conv' in lkey or 'downsample' in lkey or 'shortcut' in lkey:
		fan_in = dims[1]*dims[2]*dims[3]*connectivity # for conv layer
	elif 'fc' in lkey or 'cl' in lkey or 'linear' in lkey:
		fan_in = dims[1]*connectivity # for fc layer
	else:
		print('<!> Can not compute fan-in - unknown layer type.')
	bound = 1 / np.sqrt(fan_in)
	return fan_in, bound




def adjust_layer_init(net, lname, lname_for_statedict, tensor_dims_layer, nwtf_layer, num_W_layer):
    """ Adjust initial values of weights and biases in a sparse layer. """
    lctvt= 1-nwtf_layer/num_W_layer
    fan_in, bound = get_fanin(lname, tensor_dims_layer, lctvt)

    net.state_dict()[lname_for_statedict].data.uniform_(-bound, bound)

    lname_for_statedict_bias= lname_for_statedict.replace('weight', 'bias')
    if lname_for_statedict_bias in net.state_dict().keys():
        net.state_dict()[lname_for_statedict_bias].data.uniform_(-bound, bound)


def mask_tensor(tensor, layer_mask, io_only):
    """ Apply sparsity mask to given tensor (a layer or its gradients). """
    layer_dims = list(tensor.shape)
    if io_only:
        if len(layer_dims)==4:
            tensor.masked_fill_(layer_mask.unsqueeze(2).unsqueeze(3), 0)
        elif len(layer_dims)==2:
            tensor.masked_fill_(layer_mask, 0)
        else:
            print('<!> Number of tensor dims is not 2 and not 4!')
    else:
        tensor.masked_fill_(layer_mask, 0)



def make_smask_for_layer(num_W_layer, nwtf_layer, tensor_dims_layer, io_only):
	""" 
	Create a mask with randomly distributed 1 and 0 for a layer with given dims. 
	The mask is created as a boolean torch tensor, directly on GPU (if available).
	
    Parameters
	  num_W_layer (int) - number of weights in this layer.
	  nwtf_layer (int) - number of weights to freeze in this layer.
	  tensor_dims_layer (list of int) - layer's tensor dimensions.
	  io_only (bool) - if True, only the io dims (the first two dims of a conv layer) are sparsified,
	else the mask is uniform in all dimensions (i.e., all tensor_dims are sparsified) 

    Returns
      smask_for_layer (torch tensor, bool) - a mask for the layer with boolean values, "True" indicating that the corresponding weight will be frozen (set to zero).
	"""
	if io_only:
		if len(tensor_dims_layer)==4:
			kernel_size= np.prod(tensor_dims_layer[-2:])
			num_W_layer= int(num_W_layer/kernel_size)
			nwtf_layer= int(nwtf_layer/kernel_size)
		tensor_dims_layer= tensor_dims_layer[:2]

	# randomly generate indices of tensor elements to freeze
	inds_1d_to_freeze= random.sample( range(num_W_layer), nwtf_layer ) # 1d array
	if torch.cuda.is_available():
		smask_for_layer= torch.cuda.BoolTensor(num_W_layer).fill_(0)
	else:
		smask_for_layer= torch.BoolTensor(num_W_layer).fill_(0)
		
	smask_for_layer[inds_1d_to_freeze] = True
	smask_for_layer= smask_for_layer.view(tensor_dims_layer) # reshape to tensor dims
	return smask_for_layer



