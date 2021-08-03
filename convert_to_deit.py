#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MoCo Pre-Traind Model to DEiT')
    parser.add_argument('--input', default='', type=str, metavar='PATH', required=True,
                        help='path to moco pre-trained checkpoint')
    parser.add_argument('--output', default='', type=str, metavar='PATH', required=True,
                        help='path to output checkpoint in DEiT format')
    args = parser.parse_args()
    print(args)

    # load input
    checkpoint = torch.load(args.input, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
            # remove prefix
            state_dict[k[len("module.base_encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    # make output directory if necessary
    output_dir = os.path.dirname(args.output)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # save to output
    torch.save({'model': state_dict}, args.output)
