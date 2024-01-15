/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#pragma once
#include <torch/extension.h>

at::Tensor dcnv4_cuda_forward(
    const at::Tensor &value,
    const at::Tensor &p_offset,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int group, const int group_channels,
    const float offset_scale, const int im2col_step, const int remove_center,
    const int d_stride, const int block_thread, const bool softmax);

std::vector<at::Tensor>
dcnv4_cuda_backward(
    const at::Tensor &value,
    const at::Tensor &p_offset,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int group, const int group_channels,
    const float offset_scale, const int im2col_step, const at::Tensor &grad_output, 
    const int remove_center, const int d_stride, const int block_thread, 
    const bool softmax);