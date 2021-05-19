"""
MIT License

Copyright (c) 2020 Mahdi S. Hosseini and Mathieu Tuli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from pathlib import Path
import sys
from typing import Any, Optional, Union

import torchvision
import torch
from torchvision import transforms

mod_name = vars(sys.modules[__name__])['__name__']

if 'adas.' in mod_name:
    from .datasets import ImageNet, TinyImageNet, ADPDataset, MHIST
else:
    from datasets import ImageNet, TinyImageNet, ADPDataset, MHIST

# from .folder2lmdb import ImageFolderLMDB


def get_data(
        name: str, root: Path,
        mini_batch_size: int,
        num_workers: int,
        transform_train: transforms, 
        transform_test: transforms,
        level: str = "L3", 
        dist: bool = False) -> \
        [Any, Optional[Any], Any, Union[int, Any]]:
    if name == 'MHIST':
        num_classes = 2
        trainset = MHIST(
            root=str(root), split='train',
            transform=transform_train)
        train_sampler = \
            torch.utils.data.distributed.DistributedSampler(
                trainset) if dist else None
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True,
            sampler=train_sampler)
        testset = MHIST(
            root=str(root), split='test',
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=mini_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    elif name == 'CIFAR100':
        num_classes = 100
        trainset = torchvision.datasets.CIFAR100(
            root=str(root), train=True, download=True,
            transform=transform_train)
        train_sampler = \
            torch.utils.data.distributed.DistributedSampler(
                trainset) if dist else None
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True,
            sampler=train_sampler)
        testset = torchvision.datasets.CIFAR100(
            root=str(root), train=False,
            download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=mini_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    elif name == 'CIFAR10':
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(
            root=str(root), train=True, download=True,
            transform=transform_train)
        train_sampler = \
            torch.utils.data.distributed.DistributedSampler(
                trainset) if dist else None
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True,
            sampler=train_sampler)

        testset = torchvision.datasets.CIFAR10(
            root=str(root), train=False,
            download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=mini_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    elif name == 'ImageNet':
        num_classes = 1000
        trainset = ImageNet(
            root=str(root), split='train', download=None,
            transform=transform_train)
        # trainset = torchvision.datasets.ImageFolder(
        #     root=str(root / 'train'),
        #     transform=transform_train)
        train_sampler = \
            torch.utils.data.distributed.DistributedSampler(
                trainset) if dist else None
        # trainset = ImageFolderLMDB(str(root / 'train.lmdb'),
        #                            transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True, sampler=train_sampler)

        # testset = torchvision.datasets.ImageFolder(
        #     root=str(root / 'val'),
        #     transform=transform_test)
        testset = ImageNet(
            root=str(root), split='val', download=None,
            transform=transform_test)
        # testset = ImageFolderLMDB(str(root / 'val.lmdb'),
        #                           transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=mini_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    elif name == 'TinyImageNet':
        num_classes = 200
        trainset = TinyImageNet(
            root=str(root), split='train', download=False,
            transform=transform_train)
        # trainset = torchvision.datasets.ImageFolder(
        #     root=str(root / 'train'),
        #     transform=transform_train)
        train_sampler = \
            torch.utils.data.distributed.DistributedSampler(
                trainset) if dist else None
        # trainset = ImageFolderLMDB(str(root / 'train.lmdb'),
        #                            transform_train)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=mini_batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True, sampler=train_sampler)

        # testset = torchvision.datasets.ImageFolder(
        #     root=str(root / 'val'),
        #     transform=transform_test)
        testset = TinyImageNet(
            root=str(root), split='val', download=False,
            transform=transform_test)
        # testset = ImageFolderLMDB(str(root / 'val.lmdb'),
        #                           transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=mini_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    
    elif name == 'ADP-Release1':       
        
        num_classes = ADPDataset.ADP_classes[level]['numClasses']
        
        train_set = ADPDataset(level,
                               transform = transform_train,
                               root = str(root),
                               split = 'train')

        train_sampler = torch.utils.data.distributed.DistributedSampler(
                            train_set) if dist else None

        train_loader = torch.utils.data.DataLoader(
                            train_set, 
                            batch_size = mini_batch_size,
                            shuffle=(train_sampler is None), 
                            pin_memory = True,
                            num_workers = num_workers,
                            sampler = train_sampler)

        test_set = ADPDataset(level,
                              transform = transform_test,
                              root = str(root),
                              split = 'valid') # USING VALIDATION DATA

        test_loader = torch.utils.data.DataLoader(
                        test_set, 
                        batch_size = mini_batch_size, 
                        pin_memory = True, 
                        shuffle = False,
                        num_workers = num_workers)

    return train_loader, train_sampler, test_loader, num_classes
