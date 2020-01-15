import torch
import torchvision
import torchvision.datasets as datasets
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler, RandomSampler
import torchvision.transforms as transforms
import numpy as np


train_transforms = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomCrop(96),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def get_dataset_all(transform=train_transforms):
    dataset_unlabel = datasets.STL10('.', split='unlabeled', transform=transform, download=False)
    return dataset_unlabel


def get_train_label_dataset(transform=train_transforms):
    train_dataset_label = datasets.STL10('.', split='train', transform=transform, download=False)
    return train_dataset_label


def get_test_dataset(transform=test_transforms):
    test_dataset_label = datasets.STL10('.', split='test', transform=transform, download=False)
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


def get_loader(dataset, batch_size=64, sampler=None):
    if sampler is None:
        loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    else:
        loader = data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8)
    return loader


if __name__ == "__main__":
    unlabeled = get_dataset_all()
    train_dataset = get_train_label_dataset()
    test_dataset = get_test_dataset()

    train_loader = get_loader(train_dataset)
    for i, (img, target) in enumerate(train_loader):
        print("Train: Batch No.{}: ({}, {})".format(i, type(img), type(target)))
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