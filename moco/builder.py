# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a base encoder, a momentum encoder
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        m: moco momentum of updating momentum encoder (default: 0.99)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        # create the encoders
        # num_classes is the hidden MLP dimension
        self.base_encoder = base_encoder(num_classes=mlp_dim, zero_init_residual=True)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim, zero_init_residual=True)

        self.base_encoder.fc = nn.Sequential(self.base_encoder.fc,
                                            nn.BatchNorm1d(mlp_dim),
                                            nn.ReLU(inplace=True), # first layer
                                            nn.Linear(mlp_dim, dim)) # second layer
        self.base_encoder.fc[0].bias.requires_grad = False # hack: not use bias as it is followed by BN
        self.momentum_encoder.fc = nn.Sequential(self.momentum_encoder.fc,
                                            nn.BatchNorm1d(mlp_dim),
                                            nn.ReLU(inplace=True), # first layer
                                            nn.Linear(mlp_dim, dim)) # second layer

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, mlp_dim, bias=False),
                                        nn.BatchNorm1d(mlp_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(mlp_dim, dim)) # output layer

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def forward(self, im1, im2, m):
        """
        Input:
            im1: first views of images
            im2: second views of images
            m: moco momentum
        Output:
            logits, targets
        """

        # compute features
        p1 = self.predictor(self.base_encoder(im1))
        p2 = self.predictor(self.base_encoder(im2))
        # normalize
        p1 = nn.functional.normalize(p1, dim=1)
        p2 = nn.functional.normalize(p2, dim=1)

        # compute momentum features as targets
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder
            t1 = self.momentum_encoder(im1)
            t2 = self.momentum_encoder(im2)
            # normalize
            t1 = nn.functional.normalize(t1, dim=1)
            t2 = nn.functional.normalize(t2, dim=1)

            # gather all targets
            t1 = concat_all_gather(t1)
            t2 = concat_all_gather(t2)

        # compute logits
        # Einstein sum is more intuitive
        logits1 = torch.einsum('nc,mc->nm', [p1, t2]) / self.T
        logits2 = torch.einsum('nc,mc->nm', [p2, t1]) / self.T

        N = logits1.shape[0]  # batch size per GPU
        labels = torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()

        return logits1, logits2, labels.cuda()


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
