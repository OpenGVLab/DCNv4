# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import numpy as np

from DCNv4 import ext

shm_size_dict = {
    "8.0": 163000,
    "8.6": 99000,
    "8.7": 163000,
    "8.9": 99000,
    "9.0": 227000,
    "7.5": 64000,
    "7.0": 96000,
}

cuda_capability = f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"

if cuda_capability not in shm_size_dict:
    raise NotImplementedError

shm_size_cap = shm_size_dict[cuda_capability]

def factors(N):
    res = []
    for i in range(1, N+1):
        if N % i == 0:
            res.append(i)
    return res

def findspec(B, Q, G, C):
    d_stride = 8
    ms = factors(B*Q)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 512:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread

def findspec_bwd(B, Q, G, C):
    if C >= 64:
        d_stride = 2
    else:
        d_stride = 1

    ms = factors(B*Q)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 256:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread

class FlashDeformAttnFunction(Function):
    @staticmethod
    @torch.autocast("cuda", enabled=True, dtype=torch.float16)
    def forward(
        ctx, value, value_spatial_shapes, value_level_start_index,
        sampling_loc_attn, im2col_step, K=8
    ):

        ctx.im2col_step = im2col_step
        ctx.K = K
        d_stride, blockthread = findspec(value.shape[0], sampling_loc_attn.shape[1], value.shape[2], value.shape[3])
        d_stride_backward, blockthread_backward = findspec_bwd(value.shape[0], sampling_loc_attn.shape[1], value.shape[2], value.shape[3])

        ctx.d_stride_backward = d_stride_backward
        ctx.blockthread_backward = blockthread_backward

        output = ext.flash_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_loc_attn,
            ctx.im2col_step,
            K,
            d_stride,
            blockthread,
        )
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_loc_attn)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_loc_attn = ctx.saved_tensors
        grad_value, grad_sampling_loc_attn = ext.flash_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_loc_attn,
            grad_output.contiguous(),
            ctx.im2col_step,
            ctx.K,
            ctx.d_stride_backward,
            ctx.blockthread_backward,
        )

        return grad_value, None, None, grad_sampling_loc_attn, None, None
