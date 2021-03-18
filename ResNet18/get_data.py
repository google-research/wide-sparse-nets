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
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from google.cloud import storage


def get_dataset(dataset_name, data_dir, split, transform=None, imsize=None, bucket=None, **kwargs):

	if dataset_name in ['cifar10', 'cifar100', 'svhn', 'mnist', 'fashionmnist']:
		dataset = globals()[f'get_{dataset_name}'](data_dir, split, 
						transform=transform, imsize=imsize, **kwargs)
	else:
		print('Requested dataset is not supported!')

	item = dataset.__getitem__(0)[0]
	dataset.nchannels = item.size(0)
	dataset.imsize = item.size(1)

	return dataset


def get_aug(split, imsize=None, aug='large'):
	if aug=='large':
		imsize = imsize if imsize is not None else 224
		if split == 'train':
			return [transforms.RandomResizedCrop(imsize, scale=(0.2, 1.0))]
		else:
			return [transforms.Resize(round(imsize * 1.143)), transforms.CenterCrop(imsize)]
	elif aug =='small':
		imsize = imsize if imsize is not None else 32
		if split == 'train':
			return [transforms.RandomCrop(imsize, padding=round(imsize / 8)),
            		transforms.RandomHorizontalFlip()]
		else:
			return [transforms.Resize(imsize), transforms.CenterCrop(imsize)]
	else: # no aug for MNIST
		return []



def get_transform(split, normalize=None, transform=None, imsize=None, aug='large'):
	if transform is None:
		if normalize is None:
			normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		transform = transforms.Compose(get_aug(split, imsize=imsize, aug=aug)
										+ [transforms.ToTensor(), normalize])
	return transform



def download_blob(bucket_name, source_blob_name, destination_file_name):
    """
    Downloads the file "gs://bucket_name/source_blob_name"
    to a local file "destination_file_name".
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    if blob.exists():
        blob.download_to_filename(destination_file_name)
        print(f'<!> Blob {source_blob_name} downloaded to {destination_file_name}.')
        return True
    else:
        print(f'<!> Blob {source_blob_name} was not found in bucket {bucket_name}!')
        return False


def get_cifar10(data_dir, split, transform=None, imsize=None, bucket=None, **kwargs):
	transform = get_transform(split, transform=transform, imsize=imsize, aug='small')
	dataset = datasets.CIFAR10(data_dir, train=(split=='train'), transform=transform, download=True, **kwargs)
	dataset.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	dataset.num_classes= 10
	return dataset


def get_cifar100(data_dir, split, transform=None, imsize=None, bucket=None, **kwargs):
	transform = get_transform(split, transform=transform, imsize=imsize, aug='small')
	dataset = datasets.CIFAR100(data_dir, train=(split=='train'), transform=transform, download=True, **kwargs)
	dataset.num_classes= 100
	return dataset


def get_svhn(data_dir, split, transform=None, imsize=None, bucket=None, **kwargs):
	transform = get_transform(split, transform=transform, imsize=imsize, aug='small')
	dataset = datasets.SVHN(data_dir, split=split, transform=transform, download=True, **kwargs)
	dataset.classes = [f'{i}' for i in range(10)]
	dataset.num_classes= 10
	return dataset


def get_mnist(data_dir, split, transform=None, imsize=28, bucket=None, **kwargs):
	normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
	transform = get_transform(split, normalize=normalize, transform=transform, imsize=imsize, aug='none')
	dataset = datasets.MNIST(data_dir, train=(split=='train'), transform=transform, download=True, **kwargs)
	dataset.num_classes= 10
	return dataset


def get_fashionmnist(data_dir, split, transform=None, imsize=28, bucket=None, **kwargs):
	normalize = transforms.Normalize(mean=[0.2862], std=[0.3299])
	transform = get_transform(split, normalize=normalize, transform=transform, imsize=imsize, aug='small')
	dataset = datasets.FashionMNIST(data_dir, train=(split=='train'), transform=transform, download=True, **kwargs)
	dataset.num_classes= 10
	return dataset
