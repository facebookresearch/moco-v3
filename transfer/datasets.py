# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import oxford_flowers_dataset, oxford_pets_dataset


def build_transform(is_train, args):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.input_size, args.input_size), scale=(0.05, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(int((256 / 224) * args.input_size)),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    return transform_train if is_train else transform_test


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'imagenet':
        raise NotImplementedError("Only [cifar10, cifar100, flowers, pets] are supported; \
                for imagenet end-to-end finetuning, please refer to the instructions in the main README.")
    
    if args.data_set == 'imagenet':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000

    elif args.data_set == 'cifar10':
        dataset = datasets.CIFAR10(root=args.data_path,
                                    train=is_train,
                                    download=True,
                                    transform=transform)
        nb_classes = 10
    elif args.data_set == "cifar100":
        dataset = datasets.CIFAR100(root=args.data_path,
                                     train=is_train,
                                     download=True,
                                     transform=transform)
        nb_classes = 100
    elif args.data_set == "flowers":
        dataset = oxford_flowers_dataset.Flowers(root=args.data_path, 
                                     train=is_train,
                                     download=False,
                                     transform=transform)
        nb_classes = 102
    elif args.data_set == "pets":
        dataset = oxford_pets_dataset.Pets(root=args.data_path,
                                     train=is_train,
                                     download=False,
                                     transform=transform)
        nb_classes = 37
    else:
        raise NotImplementedError("Only [cifar10, cifar100, flowers, pets] are supported; \
                for imagenet end-to-end finetuning, please refer to the instructions in the main README.")

    return dataset, nb_classes
