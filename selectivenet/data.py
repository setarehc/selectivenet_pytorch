import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

from collections import namedtuple

import torch
import torchvision

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

class ConcreteDataset(Dataset):
    def __init__(self, root, train, normalize):
        """
        Args:
            root (string): path to xls file
            train (bool): indicates whether train or test
            transform: Optional transform to be applied on a sample.
        """
        # Read the csv file
        data_df = pd.read_excel(os.path.join(root, 'ccs.xls'))  
        data_df.columns = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'age', 'target']
        # Last column contains target
        target = data_df.target
        # First 8 columns contain features
        features = data_df.drop('target',axis=1)
        # Normalize features
        if normalize:
            features = (features-features.mean()) / features.std()
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.9, test_size=0.1, random_state=1)
        
        if train:
            self.X = X_train.to_numpy()
            self.Y = y_train.to_numpy()
        else:
            self.X = X_test.to_numpy()
            self.Y = y_test.to_numpy()
        
    def __getitem__(self, index):
        X = self.X[index]
        y = self.Y[index]
        return (X, y)

    def __len__(self):
        return len(self.X)


class DatasetBuilder(object):
    # tuple for dataset config
    DC = namedtuple('DatasetConfig', ['mean', 'std', 'input_size', 'num_classes'])
    
    DATASET_CONFIG = {
        'svhn' :   DC([0.43768210, 0.44376970, 0.47280442], [0.19803012, 0.20101562, 0.19703614], 32, 10),
        'cifar10': DC([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784], 32, 10),
        'ccs': DC([-1.838e-17, -1.118e-15,  1.267e-15,  4.0237e-16, 4.590e-16, -2.335e-16, -2.920e-16,  1.535e-16],
                  [1.00048579, 1.00048579, 1.00048579, 1.00048579, 1.00048579, 1.00048579, 1.00048579, 1.00048579], 
                  8, None),
    } 

    def __init__(self, name:str, root_path:str):
        """
        Args
        - name: name of dataset
        - root_path: root path to datasets
        """
        if name not in self.DATASET_CONFIG.keys():
            raise ValueError('name of dataset is invalid')
        self.name = name
        self.root_path = os.path.join(root_path, self.name)

    def __call__(self, train:bool, normalize:bool, augmentation=None):
        input_size = self.DATASET_CONFIG[self.name].input_size
        if self.name == 'svhn':
            transform = self._get_transform(self.name, input_size, train, normalize, augmentation)
            dataset = torchvision.datasets.SVHN(root=self.root_path, split='train' if self.train else 'test', transform=transform, download=True)
        elif self.name == 'cifar10':
            transform = self._get_transform(self.name, input_size, train, normalize, augmentation)
            dataset = torchvision.datasets.CIFAR10(root=self.root_path, train=train, transform=transform, download=True)
        elif self.name == 'ccs':
            dataset = ConcreteDataset(root=self.root_path, train=train, normalize=normalize)
        else: 
            raise NotImplementedError 

        return dataset

    def _get_transform(self, name:str, input_size:int, train:bool, normalize:bool, augmentation:str):
        transform = []
        # arugmentation
        if train:
            if augmentation == 'original':
                transform.extend([
                    torchvision.transforms.RandomHorizontalFlip(),
                ])
            elif augmentation == 'tf':
                transform.extend([
                    torchvision.transforms.RandomRotation(degrees=15),
                    torchvision.transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=None, shear=None, resample=False, fillcolor=0),
                    torchvision.transforms.RandomHorizontalFlip(),
                ])
            elif augmentation == 'lili':
                transform.extend([
                torchvision.transforms.RandomResizedCrop(input_size),
                torchvision.transforms.RandomHorizontalFlip(),
                ])
            else: raise ValueError('Incorrect augmentation type')
        else:
            pass

        # to tensor
        transform.extend([torchvision.transforms.ToTensor(),])

        # normalize
        if normalize:
            transform.extend([
                torchvision.transforms.Normalize(mean=self.DATASET_CONFIG[name].mean, std=self.DATASET_CONFIG[name].std),
            ])

        return torchvision.transforms.Compose(transform)
    
    @property
    def input_size(self):
        return self.DATASET_CONFIG[self.name].input_size

    @property
    def num_classes(self):
        return self.DATASET_CONFIG[self.name].num_classes

    def _get_mean_and_std(self):
        """
        Function that computes mean and std used in DATASET_CONFIG
        """
        #TODO: for svhn
        if self.name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(root=self.root_path, train=train, download=True)
            all_data = np.stack([np.asarray(x[0]) for x in dataset])
            mean = np.mean(all_data, axis=(0,2,3))
            std = np.std(all_data, axis=(0,2,3))
        else: 
            raise NotImplementedError 
        return mean, std
            