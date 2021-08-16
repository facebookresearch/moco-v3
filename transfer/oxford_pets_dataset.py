# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image
from typing import Any, Callable, Optional, Tuple

import numpy as np
import os
import os.path
import pickle
import scipy.io

from torchvision.datasets.vision import VisionDataset


class Pets(VisionDataset):
    
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(Pets, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        base_folder = root
        self.train = train
        annotations_path_dir = os.path.join(base_folder, "annotations")
        self.image_path_dir = os.path.join(base_folder, "images")
        
        if self.train:
            split_file = os.path.join(annotations_path_dir, "trainval.txt")
            with open(split_file) as f:
                self.images_list = f.readlines()
        else:
            split_file = os.path.join(annotations_path_dir, "test.txt")
            with open(split_file) as f:
                self.images_list = f.readlines()


    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img_name, label, species, _ = self.images_list[index].strip().split(" ")

        img_name += ".jpg"
        target = int(label) - 1

        img = Image.open(os.path.join(self.image_path_dir, img_name))
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.images_list)
