# --------------------
# Data
# --------------------

import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import torch
import random

# class AddGaussianNoise(object):
#     def __init__(self, mean=0., std=1.):
#         self.std = std
#         self.mean = mean
        
#     def __call__(self, tensor):
#         new_tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
#         return new_tensor
    
#     def __repr__(self):
#         return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
# class AddRandomNoise(object):
#     def __init__(self, ratio=0.1):
#         self.ratio = ratio
        
#     def __call__(self, tensor):
#         new_tensor = tensor + (torch.rand(tensor.size()) < self.ratio).int()
#         new_tensor = torch.clamp(new_tensor, max=1)
#         return new_tensor
    
#     def __repr__(self):
#         return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    

def prepare_train_val(train_dataset, batch_size, kwargs, validation_by):
    if validation_by == 'mnistc-mini':
        print('validation by ', validation_by)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        val_input_ims, val_ys = torch.load('../data/MNIST_C/mnistc_mini.pt')
        val_dataset = TensorDataset(val_input_ims, val_ys)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, **kwargs)
        
    elif validation_by == 'train-split':        
        n_train = int(len(train_dataset)*0.9)
        n_val = len(train_dataset)-int(len(train_dataset)*0.9)
        split_size = [n_train, n_val]
        train_set, val_set = torch.utils.data.random_split(train_dataset, split_size, generator=torch.Generator()) # torch.Generator().manual_seed(0)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
        val_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        print('validation by ', validation_by)
        print(f'# train: {n_train}, # val: {n_val}')

    return train_dataloader, val_dataloader
        
    
def fetch_dataloader(task, data_dir, device, batch_size, train=True, download=True):
    """
    load dataset depending on the task
    currently implemented tasks:
        -mnist
        -cifar10

    args
        -args
        -batch size
        -train: if True, load train dataset, else test dataset
    """
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    
    if task == 'mnist': 
        print('original mnist dataset')
        transforms = T.Compose([T.ToTensor()])
        dataset = datasets.MNIST(root=data_dir, train=train, download=download, transform=transforms,
                                target_transform=T.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
        if train: 
            train_dataloader, val_dataloader = prepare_train_val(dataset, batch_size, kwargs, validation_by='train-split')
            return train_dataloader, val_dataloader
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)

        
    elif task == 'mnist_recon': 
        data_root = '../data/MNIST_recon/'
        train_datafile = 'train_blur_k5s1.pt'#'train_clean.pt'#'train_recon_combine_x1.pt' #'train_recon_combine_x1_blot5bg.pt' 
        test_datafile= 'test_blur_k5s1.pt' #'test_clean.pt'#'test_recon_combine_x1.pt' 'test_recon_combine_x1_blot5bg.pt' 
        print(train_datafile, test_datafile)
        
        if train:
            input_ims, gt_ims, ys = torch.load(data_root+train_datafile)
            dataset = TensorDataset(input_ims, gt_ims, ys)
            train_dataloader, val_dataloader = prepare_train_val(dataset, batch_size, kwargs, validation_by='train-split')
            return train_dataloader, val_dataloader
                         
        else:
            input_ims, gt_ims, ys = torch.load(data_root+test_datafile)
            test_dataset = TensorDataset(input_ims, gt_ims, ys)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, **kwargs)
            return test_dataloader

    elif task == 'mnist_c_mini': 
        if train:
            raise NotImplementedError
        else:
            test_input_ims, test_ys = torch.load('../data/MNIST_C/mnistc_mini.pt')
            test_dataset = TensorDataset(test_input_ims, test_ys)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, **kwargs)
            return test_dataloader

        

#     elif task == 'mnist_bgimage': 
#         data_root = '../data/MNIST_bgimage/'
#         train_datafile = 'train_original_and_bgimage.pt'
#         test_datafile=  'test_original_and_bgimage.pt'
#         print(train_datafile, test_datafile)
        
#         if train:
#             input_ims, ys = torch.load(data_root+train_datafile)
#             dataset = TensorDataset(input_ims, ys)
#             train_dataloader, val_dataloader = prepare_train_val(dataset, batch_size, kwargs, validation_by='mnistc-mini')
#             return train_dataloader, val_dataloader
                         
#         else:
#             input_ims, ys = torch.load(data_root+test_datafile)
#             test_dataset = TensorDataset(input_ims, ys)
#             test_dataloader = DataLoader(test_dataset, batch_size=batch_size, **kwargs)
#             return test_dataloader

#     elif task == 'mnist_bgrandom': 
#         data_root = '../data/MNIST_bgrandom/'
#         train_datafile = 'train_original_and_bgrandom.pt'
#         test_datafile=  'test_original_and_bgrandom.pt'
#         print(train_datafile, test_datafile)
        
#         if train:
#             input_ims, ys = torch.load(data_root+train_datafile)
#             dataset = TensorDataset(input_ims, ys)
#             train_dataloader, val_dataloader = prepare_train_val(dataset, batch_size, kwargs, validation_by='mnistc-mini')
#             return train_dataloader, val_dataloader
                         
#         else:
#             input_ims, ys = torch.load(data_root+test_datafile)
#             test_dataset = TensorDataset(input_ims, ys)
#             test_dataloader = DataLoader(test_dataset, batch_size=batch_size, **kwargs)
#             return test_dataloader

        
#     elif task == 'mnist_noise': 
#         transforms = T.Compose([T.ToTensor(), AddRandomNoise(ratio=0.1)])
#         dataset = datasets.MNIST(root=data_dir, train=train, download=download, transform=transforms,
#                                 target_transform=T.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
#         if train: 
#             train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000], generator=torch.Generator().manual_seed(0))
#             train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
#             val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
#             return train_dataloader, val_dataloader
#         else:
#             return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)

#     elif task == 'mnist_blur': 
#         transforms = T.Compose([ T.GaussianBlur(5, sigma=5.0), T.ToTensor()])
#         dataset = datasets.MNIST(root=data_dir, train=train, download=download, transform=transforms,
#                                 target_transform=T.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
#         if train: 
#             train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000], generator=torch.Generator().manual_seed(0))
#             train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
#             val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
#             return train_dataloader, val_dataloader
#         else:
#             return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
        
#     elif task == 'mnist_erasing': 
#         transforms = T.Compose([T.ToTensor(), T.RandomErasing(p=1, scale=(0.1, 0.1))])
#         dataset = datasets.MNIST(root=data_dir, train=train, download=download, transform=transforms,
#                                 target_transform=T.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
#         if train: 
#             train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000], generator=torch.Generator().manual_seed(0))
#             train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
#             val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
#             return train_dataloader, val_dataloader
#         else:
#             return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)


#     elif task == 'cifar10': 

#         data_root  = data_dir + '/cifar10'    
#         #kwargs.pop('input_size', None)
#         transform = T.Compose([T.ToTensor()])  # transforms.RandomCrop(size=32, padding=shift_pixels),
#         dataset = datasets.CIFAR10(root=data_root, train=train, download=download, transform=transform,
#                                    target_transform=T.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
        
#         if train: 
#             train_set, val_set = torch.utils.data.random_split(dataset, [40000, 10000], generator=torch.Generator().manual_seed(0))
#             train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
#             val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
#             return train_dataloader, val_dataloader
        
#         else: 
#             return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)