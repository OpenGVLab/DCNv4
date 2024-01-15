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

from functions import MSDeformAttnFunction, ms_deform_attn_core_pytorch, FlashDeformAttnFunction


H = 256
W = 256
N, M, D = 1, 8, 16
Lq, L, P = H * W, 1, 8

# x = x.reshape([B, H*W, G, D + self.num_split * 3])
shapes = torch.as_tensor([(H, W)], dtype=torch.long).cuda()
# shapes = torch.as_tensor([(H, W), (H // 2, W // 2)], dtype=torch.long).cuda()
# shapes = torch.as_tensor([(H, W), (H // 2, W // 2), (H // 4, W // 4), (H // 8, W // 8)], dtype=torch.long).cuda()

H = 256
W = 256
N, M, D = 1, 8, 32
Lq, L, P = 100*152, 4, 8

shapes = torch.Tensor([[100, 152], [ 50,  76], [ 25,  38], [ 13,  19]]).long().cuda()

level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
S = sum([(H * W).item() for H, W in shapes])

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


def check_forward_equal_with_pytorch_half():
    value = torch.rand(N, S, M, D).cuda() * 0.01
    offset = (torch.rand(N, Lq, M, L, P, 2).cuda() * 2 - 1) / 10
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights_origin = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights_origin.requires_grad = True
    sampling_loc_attn = torch.cat([sampling_locations.detach().reshape(N, Lq, M, L*P*2), attention_weights_origin.detach().reshape(N, Lq, M, L*P)], dim=-1)

    attention_weights = torch.nn.functional.softmax(attention_weights_origin.flatten(-2, -1), dim=-1).unflatten(-1, (L, P))


    im2col_step = 128

    value.requires_grad = True
    sampling_loc_attn.requires_grad = True
    output_cuda = (
        FlashDeformAttnFunction.apply(
            value.float(),
            shapes,
            level_start_index,
            sampling_loc_attn.float(),
            im2col_step,
        )
    )
    (output_cuda.float().sum()/10).backward()


    value1 = value.detach()
    value1.requires_grad = True
    sampling_locations.requires_grad = True
    #attention_weights.requires_grad = True
    output_pytorch = (
        ms_deform_attn_core_pytorch(value1, shapes, sampling_locations, attention_weights)
    )
    (output_pytorch.sum()/10).backward()

    max_abs_err = (output_cuda.float() - output_pytorch).abs().max()
    max_rel_err = ((output_cuda.float() - output_pytorch).abs() / output_pytorch.abs()).max()
    fwdok = torch.allclose(output_cuda.float(), output_pytorch, rtol=1e-2, atol=1e-3)
    print(fwdok)
    print(max_abs_err, max_rel_err)
    #exit()

    bwdok1 = torch.allclose(value.grad, value1.grad, rtol=1e-2, atol=1e-3)
    print(bwdok1)
    # rel_err = (sampling_locations.grad - sampling_loc_attn.grad[..., :L*P*2].reshape(*sampling_locations.shape)).abs()/(sampling_locations.grad.abs()+1e-3)
    # print(rel_err.max())

    locgrad1 = sampling_locations.grad
    locgrad2 = sampling_loc_attn.grad[..., :L*P*2].reshape(*sampling_locations.shape)
    bwdok2 = torch.allclose(locgrad1, locgrad2, rtol=1e-2, atol=1e-3)
    print(bwdok2)
    rel_err = (locgrad1 - locgrad2).abs()/(locgrad1.abs()+1e-3)
    print(rel_err.max())

    attngrad1 = attention_weights_origin.grad
    attngrad2 = sampling_loc_attn.grad[..., L*P*2:].reshape(*attention_weights_origin.shape)
    bwdok3 = torch.allclose(locgrad1, locgrad2, rtol=1e-2, atol=1e-3)
    print(bwdok3)
    rel_err = (attngrad1 - attngrad2).abs()/(attngrad1.abs()+1e-3)
    print(rel_err.max())
    exit()
    #exit()

    # pdb.set_trace()
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()
    fwdok = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)

    print(
        f"* {fwdok} check_forward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}"
    )


    fn_args = (
        value,
        shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    )

    flash_dcn_fn_args = (
        value.half(),
        shapes,
        level_start_index,
        sampling_loc_attn.half(),
        im2col_step,
    )


    test_args = edict({'warmup_num': 50, 'test_num': 100})
    exp_time = speed_test(
            FlashMSDeformAttnFunction.apply, test_args, flash_dcn_fn_args, name='exp')
    exp_time_base = speed_test(
            MSDeformAttnFunction.apply, test_args, fn_args, name='exp')

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