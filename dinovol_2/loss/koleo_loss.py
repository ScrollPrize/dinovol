# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_functional
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("dinov2")


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self, *, distributed: bool = False, process_group=None):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)
        self.distributed = bool(distributed)
        self.process_group = process_group

    def set_process_group(self, process_group) -> None:
        self.process_group = process_group

    def _gather_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if not self.distributed or not dist.is_available() or not dist.is_initialized():
            return x
        gathered = dist_nn_functional.all_gather(x, group=self.process_group)
        return torch.cat(tuple(gathered), dim=0)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.amp.autocast(device_type=student_output.device.type, enabled=False):
            student_output = self._gather_if_needed(student_output)
            if student_output.shape[0] < 2:
                return student_output.new_zeros(())
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss
