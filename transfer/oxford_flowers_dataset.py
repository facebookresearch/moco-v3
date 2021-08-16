# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
from PIL import Image
from typing import Any, Callable, Optional, Tuple

import numpy as np
import os
import os.path
import pickle
import scipy.io

from torchvision.datasets.vision import VisionDataset


class Flowers(VisionDataset):
    
    def __init__(
            self,
            root,
            train=True,
            transform=None,
            target_transform=None,
            download=False,
    ):

        super(Flowers, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        base_folder = root
        self.image_folder = os.path.join(base_folder, "jpg")
        label_file = os.path.join(base_folder, "imagelabels.mat")
        setid_file = os.path.join(base_folder, "setid.mat") 

        self.train = train

        self.labels = scipy.io.loadmat(label_file)["labels"][0]
        train_list = scipy.io.loadmat(setid_file)["trnid"][0]
        val_list = scipy.io.loadmat(setid_file)["valid"][0]
        test_list = scipy.io.loadmat(setid_file)["tstid"][0]
        trainval_list = np.concatenate([train_list, val_list])
        
        if self.train:
          self.img_files = trainval_list
        else:
          self.img_files = test_list
          

    def __getitem__(self, index):
        img_name = "image_%05d.jpg" % self.img_files[index]
        target = self.labels[self.img_files[index] - 1] - 1
        img = Image.open(os.path.join(self.image_folder, img_name))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img_files)
