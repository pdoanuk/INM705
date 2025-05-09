# -*- coding: utf-8 -*-
## TIMESTAMP @ 2025-04-23T09:00:45
## author: phuocddat

# ATTRIBUTION NOTICE:
# Portions of this code are adapted from or reference the following sources:
# - ViTAD / ADer: https://github.com/zhangzjn/ADer/
# Significant modifications and original code added by phuocddat.
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
from typing import List, Union, Optional, Tuple
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from timm.scheduler.cosine_lr import CosineLRScheduler

def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    """Generates a 1D Gaussian kernel."""
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int = 1) -> torch.Tensor:
    """Generates a 2D Gaussian kernel window."""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# --- Loss Function Definitions ---

class L2Loss(nn.Module):
    """ Compute the L2 (MSE) loss """

    def __init__(self, lam: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)
        self.lam = lam

    def forward(self, input1: Union[torch.Tensor, List[torch.Tensor]],
                input2: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        input1_list = input1 if isinstance(input1, list) else [input1]
        input2_list = input2 if isinstance(input2, list) else [input2]

        total_loss = 0.0
        num_items = len(input1_list)
        if num_items == 0:
            return torch.tensor(0.0, device=input1[0].device if isinstance(input1, list) and input1 else (
                input1.device if torch.is_tensor(input1) else 'cpu'))

        for in1, in2 in zip(input1_list, input2_list):
            total_loss += self.loss_fn(in1, in2)

        # if reduction == 'mean': total_loss = total_loss / num_items

        return total_loss * self.lam


class CosLoss(nn.Module):
    """Calculates the Cosine Similarity loss (1 - cosine_similarity)"""
    def __init__(self, avg: bool = True, flat: bool = True, lam: float = 1.0, dim: int = 1, eps: float = 1e-8):
        super().__init__()
        self.lam = lam
        self.avg = avg
        self.flat = flat # Flatten spatial dimensions before calculation
        self.dim = dim # Dimension over which to compute cosine similarity
        self.eps = eps # Epsilon for numerical stability

    def forward(self, input1: Union[torch.Tensor, List[torch.Tensor]], input2: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        input1_list = input1 if isinstance(input1, list) else [input1]
        input2_list = input2 if isinstance(input2, list) else [input2]

        total_loss = 0.0
        num_items = len(input1_list)
        if num_items == 0:
            return torch.tensor(0.0, device=input1[0].device if isinstance(input1, list) and input1 else (input1.device if torch.is_tensor(input1) else 'cpu'))

        for in1, in2 in zip(input1_list, input2_list):
            if self.flat:
                in1_flat = in1.contiguous().view(in1.shape[0], -1)
                in2_flat = in2.contiguous().view(in2.shape[0], -1)
                # Cosine similarity applied per batch item, then mean loss
                cos_sim = F.cosine_similarity(in1_flat, in2_flat, dim=1, eps=self.eps) # Compare flattened features for each item in batch
                loss_per_pair = (1.0 - cos_sim).mean() # Average loss over batch
            else:
                # Compare tensors directly along the specified dimension
                cos_sim = F.cosine_similarity(in1.contiguous(), in2.contiguous(), dim=self.dim, eps=self.eps)
                loss_per_pair = (1.0 - cos_sim).mean() # Average loss over all elements

            total_loss += loss_per_pair * self.lam

        return total_loss / num_items if self.avg and num_items > 0 else total_loss



class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]