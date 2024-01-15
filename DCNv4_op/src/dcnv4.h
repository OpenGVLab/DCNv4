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

#ifdef WITH_CUDA
#include "cuda/dcnv4_cuda.h"
#include "cuda/flash_deform_attn_cuda.h"
#endif

at::Tensor flash_deform_attn_forward(const at::Tensor &value,
                                        const at::Tensor &spatial_shapes,
                                        const at::Tensor &level_start_index,
                                        const at::Tensor &sampling_loc_attn,
                                        const int im2col_step, const int K,
                                        const int d_stride, const int block_thread) {
  if (value.device().is_cuda()) {
#ifdef WITH_CUDA
    return flash_deform_attn_cuda_forward(value, spatial_shapes,
                                             level_start_index,
                                             sampling_loc_attn, im2col_step, 
                                             K, d_stride, block_thread);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
flash_deform_attn_backward(const at::Tensor &value, 
                              const at::Tensor &spatial_shapes,
                              const at::Tensor &level_start_index, 
                              const at::Tensor &sampling_loc_attn,
                              const at::Tensor &grad_output, 
                              const int im2col_step, 
                              const int K, 
                              const int d_stride, const int block_thread){
  if (value.device().is_cuda()) {
#ifdef WITH_CUDA
    return flash_deform_attn_cuda_backward(value, 
                                              spatial_shapes,
                                              level_start_index,
                                              sampling_loc_attn, 
                                              grad_output,
                                              im2col_step, 
                                              K, d_stride, 
                                              block_thread);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor dcnv4_forward(
    const at::Tensor &value,
    const at::Tensor &p_offset,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int group, const int group_channels,
    const float offset_scale, const int im2col_step, const int remove_center,
    const int d_stride, const int block_thread, const bool softmax) {
  if (value.device().is_cuda()) {
#ifdef WITH_CUDA
    return dcnv4_cuda_forward(
        value, p_offset, kernel_h, kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w, group, group_channels, offset_scale,
        im2col_step, remove_center, d_stride, block_thread, softmax);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
dcnv4_backward(
    const at::Tensor &value, 
    const at::Tensor &p_offset, 
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int group, const int group_channels,
    const float offset_scale, const int im2col_step, const at::Tensor &grad_output, 
    const int remove_center, const int d_stride, const int block_thread,
    const bool softmax){
  if (value.device().is_cuda()) {
#ifdef WITH_CUDA
    return dcnv4_cuda_backward(
        value, p_offset, kernel_h, kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w, group, group_channels, offset_scale,
        im2col_step, grad_output, remove_center, d_stride, block_thread,
        softmax);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}