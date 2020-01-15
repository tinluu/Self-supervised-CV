from __future__ import print_function

import torch
import torchvision
import torchvision.datasets as datasets
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, RandomSampler
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
import os


default_transform = transforms.Compose([
    # transforms.Resize((32, 32)),
    transforms.ToTensor(),
    # transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

class MySTL10(datasets.STL10):
    def __init__(self, *args, **kwargs):
        super(MySTL10, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img1, img2), target


def get_dataset_all(transform=default_transform):
    dataset_unlabel = MySTL10('..', split='unlabeled', transform=transform, download=False)
    return dataset_unlabel


def get_train_label_dataset(transform=default_transform):
    train_dataset_label = datasets.STL10('..', split='train', transform=transform, download=False)
    return train_dataset_label


def get_test_dataset(transform=default_transform ):
    test_dataset_label = datasets.STL10('..', split='test', transform=transform, download=False)
    return test_dataset_label 


def get_sub_sampler(dataset, ratio=1.0):
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor((ratio) * len(indices)))
    np.random.shuffle(indices)
    indices = indices[:split]
    sampler = SubsetRandomSampler(indices)
    return sampler


def train_val_sampler(dataset, ratio=1.0, train_ratio=0.95):
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor((ratio) * len(indices)))
    np.random.shuffle(indices)
    indices = indices[:split]
    split = int(np.floor((train_ratio) * len(indices)))
    train_ids, val_ids = indices[:split], indices[split:]
    train_sampler, val_sampler = SubsetRandomSampler(train_ids), SubsetRandomSampler(val_ids)
    return train_sampler, val_sampler


def get_loader(dataset, batch_size=64, sampler=None, num_workers=8):
    if sampler is None:
        loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        loader = data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    return loader


if __name__ == "__main__":
    unlabeled = get_dataset_all(transform=default_transform )
    train_dataset = get_train_label_dataset(transform=default_transform )
    test_dataset = get_test_dataset(transform=default_transform )

    train_loader = get_loader(train_dataset)
    for i, (img, target) in enumerate(train_loader):
        print("Train: Batch No.{}: ({}, {})".format(i, type(img), type(target)))
        print(target)
        break

    test_loader = get_loader(test_dataset)
    for i, (img, target) in enumerate(test_loader):
        print("Test: Batch No.{}: ({}, {})".format(i, type(img), type(target)))
        break
    
    train_sampler, val_sampler = train_val_sampler(unlabeled)

    unlabeled_train_loader = get_loader(unlabeled, batch_size=1, sampler=train_sampler)
    unlabeled_val_loader = get_loader(unlabeled, batch_size=1, sampler=val_sampler)

    print("Unlabeled data size: train {}; val {}".format(len(unlabeled_train_loader), len(unlabeled_val_loader)))

    for i, (img, target) in enumerate(unlabeled_train_loader):
        print("Unlabeled_all: Batch No.{}: ({}, {})".format(i, img.size(), target))
        break

    for i, (img, target) in enumerate(unlabeled_val_loader):
        print("Unlabeled_all: Batch No.{}: ({}, {})".format(i, img.size(), target))
        break