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

#include "cuda/flash_deform_im2col_cuda.cuh"
#include "cuda/flash_deform_col2im_cuda.cuh"
#include <vector>

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

at::Tensor flash_deform_attn_cuda_forward(
    const at::Tensor &value, const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index, const at::Tensor &sampling_loc_attn,
    const int im2col_step = 64, const int K=8, const int d_stride=8,
    const int block_thread=0) {
  AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
  AT_ASSERTM(spatial_shapes.is_contiguous(),
             "spatial_shapes tensor has to be contiguous");
  AT_ASSERTM(level_start_index.is_contiguous(),
             "level_start_index tensor has to be contiguous");
  AT_ASSERTM(sampling_loc_attn.is_contiguous(),
             "sampling_loc_attn tensor has to be contiguous");

  AT_ASSERTM(value.type().is_cuda(), "value must be a CUDA tensor");
  AT_ASSERTM(spatial_shapes.type().is_cuda(),
             "spatial_shapes must be a CUDA tensor");
  AT_ASSERTM(level_start_index.type().is_cuda(),
             "level_start_index must be a CUDA tensor");
  AT_ASSERTM(sampling_loc_attn.type().is_cuda(),
             "sampling_loc_attn must be a CUDA tensor");

  const int batch = value.size(0);
  const int spatial_size = value.size(1);
  const int num_heads = value.size(2);
  const int num_channels = value.size(3);

  const int num_levels = spatial_shapes.size(0);
  const int num_query = sampling_loc_attn.size(1);
  const int num_point = K;

  const int im2col_step_ = std::min(batch, im2col_step);
  AT_ASSERTM(batch % im2col_step_ == 0, "batch(", batch,
             ") must divide im2col_step(", im2col_step_, ")");

  auto output =
      at::zeros({batch, num_query, num_heads, num_channels}, value.options());

  auto per_value_size = spatial_size * num_heads * num_channels;
  auto per_offset_size = num_query * num_heads * num_levels * num_point * 3;
  auto per_out_size = num_query * num_heads * num_channels;

  for (int n = 0; n < batch / im2col_step_; ++n) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, value.scalar_type(),
        "flash_deform_attn_forward_cuda", ([&] {
          flash_deformable_im2col_cuda(
              at::cuda::getCurrentCUDAStream(),
              value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
              spatial_shapes.data<int64_t>(), level_start_index.data<int64_t>(),
              sampling_loc_attn.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_offset_size,
              output.data_ptr<scalar_t>() + n * im2col_step_ * per_out_size,
              im2col_step_, spatial_size, num_heads, num_channels, num_levels,
              num_query, num_point, d_stride, block_thread, true);
        }));
  }
  output = output.view({batch, num_query, num_heads * num_channels});
  return output;
}

std::vector<at::Tensor>
flash_deform_attn_cuda_backward(
    const at::Tensor &value, const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index, const at::Tensor &sampling_loc_attn,
    const at::Tensor &grad_output, const int im2col_step = 64, const int K=8,
    const int d_stride=2, const int block_thread=0) {
  AT_ASSERTM(value.is_contiguous(), "value tensor has to be contiguous");
  AT_ASSERTM(spatial_shapes.is_contiguous(),
             "spatial_shapes tensor has to be contiguous");
  AT_ASSERTM(level_start_index.is_contiguous(),
             "level_start_index tensor has to be contiguous");
  AT_ASSERTM(sampling_loc_attn.is_contiguous(),
             "sampling_loc_attn tensor has to be contiguous");
  AT_ASSERTM(grad_output.is_contiguous(),
             "grad_output tensor has to be contiguous");

  AT_ASSERTM(value.type().is_cuda(), "value must be a CUDA tensor");
  AT_ASSERTM(spatial_shapes.type().is_cuda(),
             "spatial_shapes must be a CUDA tensor");
  AT_ASSERTM(level_start_index.type().is_cuda(),
             "level_start_index must be a CUDA tensor");
  AT_ASSERTM(sampling_loc_attn.type().is_cuda(),
             "sampling_loc_attn must be a CUDA tensor");
  AT_ASSERTM(grad_output.type().is_cuda(),
             "grad_output must be a CUDA tensor");

  const int batch = value.size(0);
  const int spatial_size = value.size(1);
  const int num_heads = value.size(2);
  const int num_channels = value.size(3);

  const int num_levels = spatial_shapes.size(0);
  const int num_query = sampling_loc_attn.size(1);
  const int num_point = K;

  const int im2col_step_ = std::min(batch, im2col_step);
  AT_ASSERTM(batch % im2col_step_ == 0, "batch(", batch,
             ") must divide im2col_step(", im2col_step_, ")");

  auto dtype = value.dtype();
  if (dtype == at::kHalf){
    dtype = at::kFloat;
  }

  auto grad_input = at::zeros_like(value, dtype);
  auto grad_offset = at::zeros_like(sampling_loc_attn, dtype);

  auto per_value_size = spatial_size * num_heads * num_channels;
  auto per_offset_size = num_query * num_heads * num_levels * num_point * 3;
  auto per_out_size = num_query * num_heads * num_channels;

  for (int n = 0; n < batch / im2col_step_; ++n) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, value.scalar_type(),
        "flash_deform_attn_backward_cuda", ([&] {
          flash_deformable_col2im_cuda(
              at::cuda::getCurrentCUDAStream(),
              value.data_ptr<scalar_t>() + n * im2col_step_ * per_value_size,
              spatial_shapes.data<int64_t>(), level_start_index.data<int64_t>(),
              sampling_loc_attn.data_ptr<scalar_t>() +
                  n * im2col_step_ * per_offset_size,
              grad_output.data_ptr<scalar_t>() + n * im2col_step_ * per_out_size,
              im2col_step_, spatial_size, num_heads, num_channels, num_levels,
              num_query, num_point,
              grad_input.data<opmath_t>() + n * im2col_step_ * per_value_size,
              grad_offset.data<opmath_t>() + n * im2col_step_ * per_offset_size,
              d_stride, block_thread
              );
        }));
  }

  if(value.dtype() == torch::kHalf){
    return {grad_input.to(torch::kHalf), grad_offset.to(torch::kHalf)};
  }
  else{
    return {grad_input, grad_offset};
  }
}
