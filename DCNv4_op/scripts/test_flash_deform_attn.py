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
from easydict import EasyDict as edict
from torch.cuda import Event
import pandas as pd

import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from functions import MSDeformAttnFunction, FlashDeformAttnFunction, ms_deform_attn_core_pytorch


# N, M, D = 1, 4, 8
# # Lq, L, P = 2, 2, 2
# # shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
# Lq, L, P = 1, 2, 8
# shapes = torch.as_tensor([(8, 16), (4, 8)], dtype=torch.long).cuda()

# N, M, D = 1, 8, 32
# # Lq, L, P = 2, 2, 2
# # shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
# Lq, L, P = 300, 4, 4
# # shapes = torch.as_tensor([(134, 151), (67, 76), (34, 38), (17, 19)], dtype=torch.long).cuda()
# # shapes = torch.as_tensor([(134, 151), (67, 76), (34, 38), (16, 16)], dtype=torch.long).cuda()
# # shapes = torch.as_tensor([(134, 151), (67, 76), (34, 38), (17, 19)], dtype=torch.long).cuda()
# # shapes = torch.as_tensor([(17, 19), (4, 4)], dtype=torch.long).cuda()
# shapes = torch.as_tensor([(100, 151), (50, 76), (25, 38), (13, 19)], dtype=torch.long).cuda()
# # shapes = torch.as_tensor([(110, 151)], dtype=torch.long).cuda()

# B:6
# H:232
# W:400
# G:5
# D: 16
# channels: 80
# kernel: 3 points = 3 * 3
# num_split = 45 = kernel *kernel * G

H = 256
W = 256
N, M, D = 1, 8, 32
Lq, L, P = 100*152, 4, 8

shapes = torch.Tensor([[100, 152], [ 50,  76], [ 25,  38], [ 13,  19]]).long().cuda()

# x = x.reshape([B, H*W, G, D + self.num_split * 3])
# shapes = torch.as_tensor([(H, W)], dtype=torch.long).cuda()
# shapes = torch.as_tensor([(H, W), (H // 2, W // 2)], dtype=torch.long).cuda()
# shapes = torch.as_tensor([(H, W), (H // 2, W // 2), (H // 4, W // 4), (H // 8, W // 8)], dtype=torch.long).cuda()

level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
S = sum([(H * W).item() for H, W in shapes])
print(S)

def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):

        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                        torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / (H_)
        ref_x = ref_x.reshape(-1)[None] / (W_)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    # reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points


torch.manual_seed(3)

@torch.no_grad()
def speed_test(func, args, inputs, name='Unknown'):

    tic = Event(enable_timing=True)
    toc = Event(enable_timing=True)
    # warmup
    for i in range(args.warmup_num):
        func(*inputs)

    tic.record()
    for i in range(args.test_num):
        func(*inputs)
    toc.record()
    torch.cuda.synchronize()

    avg_time = tic.elapsed_time(toc) / args.test_num
    print(
        f'>>> {name: <10} finished {args.test_num} running, avg_time: {avg_time:.6f} ms')
    return avg_time


@torch.no_grad()
def check_forward_equal_with_pytorch_half():
    value = torch.rand(N, S, M, D).cuda() * 0.01
    # offset = (torch.rand(N, Lq, M, L, P, 2).cuda() * 2 - 1) / 10
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    sampling_loc_attn = torch.cat([sampling_locations.reshape(N, Lq, M, L*P*2), attention_weights.reshape(N, Lq, M, L*P)], dim=-1)
    attention_weights = torch.nn.functional.softmax(attention_weights.flatten(-2, -1), dim=-1).unflatten(-1, (L, P))


    im2col_step = 128

    flash_fn_args = (
        value.half(),
        shapes,
        level_start_index,
        sampling_loc_attn.half(),
        im2col_step,
        P, 16
    )
    output_cuda = (
        FlashDeformAttnFunction.apply(*flash_fn_args)
        .detach()
        .cpu()
    ).double()
    
    fn_args = (
        value,
        shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    )

    output_pytorch = (
        MSDeformAttnFunction.apply(*fn_args)
        .detach().double()
        .cpu()
    )

    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()
    fwdok = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)

    print(
        f"* {fwdok} check_forward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}"
    )

    test_args = edict({'warmup_num': 1000, 'test_num': 1000})
    exp_time_base = speed_test(
            MSDeformAttnFunction.apply, test_args, fn_args, name='exp')
    exp_time = speed_test(
            FlashDeformAttnFunction.apply, test_args, flash_fn_args, name='exp')

    results = [{}]
    results[0]['time'] = exp_time
    results[0]['time_base'] = exp_time_base
    columns = list(results[0].keys())

    outputs = pd.DataFrame(results, columns=columns)
    with pd.option_context(
        'display.max_rows', None, 'display.max_columns', None,
        'display.max_colwidth', None, 'display.width', None,
        'display.precision', 4, ):
        print(outputs)


if __name__ == "__main__":
    check_forward_equal_with_pytorch_half()

