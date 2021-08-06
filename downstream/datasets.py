import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import oxford_flowers_dataset, oxford_pets_dataset


def build_other_transform(is_train, args):
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


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_dataset(is_train, args):
    if args.data_set == 'imagenet':
        transform = build_transform(is_train, args)
    else:
        transform = build_other_transform(is_train, args)
    
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
        raise NotImplementedError

    return dataset, nb_classes
