# --------------------------------------------------------
# DCNv4
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import torch
import torch.nn as nn
import math
from torch.autograd import gradcheck
import pandas as pd
from easydict import EasyDict as edict
import argparse

from torch.cuda import Event

from functions import DCNv4Function, DCNv3Function
torch.set_printoptions(threshold=10000)



torch.manual_seed(3)

def speed_test_backward(func, args, inputs, name='Unknown'):
    # warmup
    # for i in range(args.warmup_num):
    #     o = func(*inputs)
    #     o.sum().backward()

    total_time = 0
    len_input = len(inputs)
    for i in range(args.warmup_num + args.test_num):
        tic = Event(enable_timing=True)
        toc = Event(enable_timing=True)
        inputs[0] = inputs[0].detach()
        inputs[0].requires_grad = True
        if len_input > 1 and isinstance(inputs[1], torch.Tensor):
            inputs[1] = inputs[1].detach()
            inputs[1].requires_grad = True
        if len_input > 2 and isinstance(inputs[2], torch.Tensor):
            inputs[2] = inputs[2].detach()
            inputs[2].requires_grad = True

        o = func(*inputs)
        torch.cuda.synchronize()
        tic.record()
        o.sum().backward()
        toc.record()
        torch.cuda.synchronize()
        _time = tic.elapsed_time(toc)
        if i >= args.warmup_num:
            total_time += _time
        o = o.detach()

    # toc.record()
    # torch.cuda.synchronize()

    avg_time = total_time / args.test_num
    #print(
    #    f'>>> {name: <10} finished {args.test_num} running, avg_time: {avg_time:.6f} ms')
    return avg_time

# @torch.no_grad()
def test(N=64, H_in=32, W_in=32, M=4, D=16, spec=None):
    """
    64x56x56x128(G=4)
    2 64: 3.66
    - offset_mask collection write 3.4022
    - offset_mask collection 3.1968
    
    """
    Kh, Kw = 3, 3
    remove_center = False
    P = Kh * Kw - remove_center
    offset_scale = 2.0
    pad = 1
    dilation = 1
    stride = 1
    H_out = (H_in + 2 * pad - (dilation * (Kh - 1) + 1)) // stride + 1
    W_out = (W_in + 2 * pad - (dilation * (Kw - 1) + 1)) // stride + 1

    additions = [None, None, spec[0], spec[1], False]
    input = torch.rand(N, H_in, W_in, M*D).cuda() * 10
    #offset = torch.rand(N, H_out, W_out, M*P*2).cuda() * 0
    offset = (torch.rand(N, H_out, W_out, M*P*2).cuda() * 2 - 1)*2
    mask_origin = torch.rand(N, H_out, W_out, M, P).cuda() + 1e-5
    mask_origin = mask_origin.half()
    mask_origin.requires_grad = True
    # offset_mask = torch.cat([offset.unflatten(-1, (M, P, 2)), mask_origin.detach().unsqueeze(-1)], dim=-1).flatten(-3)
    # mask /= mask.sum(-1, keepdim=True)
    # mask = torch.nn.functional.softmax(mask_origin, dim=-1, dtype=torch.float32)
    mask = mask_origin
    # mask = mask.reshape(N, H_out, W_out, M*P)
    # offset_mask = torch.cat([offset.unflatten(-1, (M, P, 2)), mask.detach().unsqueeze(-1)], dim=-1).flatten(-3)
    offset_mask = torch.cat([offset.detach().unflatten(-1, (M, P * 2)), mask_origin.detach()], dim=-1).flatten(-2)

    im2col_step = 128

    input = input.half()
    offset = offset.half()
    mask = mask.half()
    input.requires_grad = True
    offset.requires_grad = True
    # mask.requires_grad = True
    output_pytorch = DCNv3Function.apply(
        input,
        offset,
        mask,
        Kh, Kw, stride, stride, Kh // 2, Kw // 2, dilation, dilation, M, D, offset_scale,
        im2col_step, remove_center)#.detach().cpu()
    (output_pytorch.sum()/10).backward()

    def pad(om):
        padded_zero = int(math.ceil(om.shape[3]/8)*8) - om.shape[3]
        padded = torch.zeros(om.shape[0], om.shape[1], om.shape[2], padded_zero).to(om)
        return torch.cat([om, padded], dim=-1)

    # value_offset_mask = input.detach()
    input1 = input.detach()
    input1.requires_grad = True
    offset_mask = offset_mask.half()
    offset_mask.requires_grad = True
    # offset_mask1.requires_grad = True
    torch.cuda.profiler.cudart().cudaProfilerStart()
    output_flash_cuda = DCNv4Function.apply(
        input1, offset_mask,
        Kh, Kw, stride, stride, Kh // 2, Kw // 2, dilation, dilation, M, D, offset_scale,
        im2col_step, remove_center, *additions)#.detach().cpu()
    (output_flash_cuda.sum()/10).backward()
    torch.cuda.profiler.cudart().cudaProfilerStop()

    input_grad = input.grad
    input2_grad = input1.grad
    bwdok = torch.allclose(input_grad.float(), input2_grad.float(), rtol=1e-2, atol=1e-3)
    rel_err = (input_grad.abs() - input2_grad.abs())/(input_grad.abs()+1e-3)

    offset_grad1 = offset.grad
    offset_grad2 = offset_mask.grad.reshape(N, H_out, W_out, M, P*3)[..., :P*2].reshape(N, H_out, W_out, M*P*2)

    bwdok2 = torch.allclose(offset_grad1.float(), offset_grad2.float(), rtol=1e-2, atol=1e-3)
    rel_err = (offset_grad1 - offset_grad2).abs() / (offset_grad1.abs()+1e-3)

    mask_grad1 = mask_origin.grad
    mask_grad2 = offset_mask.grad.reshape(N, H_out, W_out, M, P*3)[..., P*2:].reshape(N, H_out, W_out, M, P)

    bwdok3 = torch.allclose(mask_grad1, mask_grad2, rtol=1e-2, atol=1e-3)
    rel_err = (mask_grad1 - mask_grad2).abs() / (mask_grad1.abs()+1e-3)

    fwdok = torch.allclose(output_flash_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_flash_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_flash_cuda - output_pytorch).abs() /
                   (output_pytorch.abs()+ 1e-3)).max()
    if not (bwdok and bwdok2 and bwdok3):
        print(f"Wrong: {N}x{H_in}x{W_in}x{M}x{D} \t {spec[0]}/{spec[1]}({spec[2]})")
        return
    # fn_args = [
    #     input,
    #     offset,
    #     mask,
    #     Kh, Kw, stride, stride, Kh // 2, Kw // 2, dilation, dilation, M, D, offset_scale,
    #     im2col_step, remove_center
    # ]

    flash_dcn_fn_args = [
        input1,
        offset_mask,
        Kh, Kw, stride, stride, Kh // 2, Kw // 2, dilation, dilation, M, D, offset_scale,
        im2col_step, remove_center, *additions
    ]

    test_args = edict({'warmup_num': 1000, 'test_num': 1000})
    try:
        exp_time = speed_test_backward(DCNv4Function.apply, test_args, flash_dcn_fn_args, name='exp')
    except:
        print(f"Wrong: {N}x{H_in}x{W_in}x{M}x{D} \t {spec[0]}/{spec[1]}({spec[2]})")
        return

    torch.cuda.synchronize()
    print(f"{N}x{H_in}x{W_in}x{M}x{D} \t {spec[0]}/{spec[1]}({spec[2]}): {exp_time}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int)
    parser.add_argument("--h", type=int)
    parser.add_argument("--w", type=int)
    parser.add_argument("--g", type=int)
    parser.add_argument("--c", type=int)
    parser.add_argument("--dstride", type=int)
    parser.add_argument("--blockthread", type=int)
    parser.add_argument("--multiplier", type=int)
    args = parser.parse_args()
    test(args.n, args.h, args.w, args.g, args.c, (args.dstride, args.blockthread, args.multiplier))


