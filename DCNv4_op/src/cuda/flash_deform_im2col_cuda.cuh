/*!
**************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************
* Modified from DCN (https://github.com/msracver/Deformable-ConvNets)
* Copyright (c) 2018 Microsoft
**************************************************************************
*/
#include <algorithm>
#include <cstdio>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THCAtomics.cuh>

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "common.h"

template <typename scalar_t, int d_stride, typename transfer_t, int L, int K>
__global__ void
forward_kernel(const scalar_t *p_value, const int64_t *data_spatial_shapes,
               const int64_t *data_level_start_index, const scalar_t *p_offset,
               scalar_t *p_output, const int N, const int G, const int D,
               const int Q, const int block_multiplier) {

  extern __shared__ char _s[];

  const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
  const int &bi = blockIdx.x * block_multiplier / Q;

  const int &di_s = threadIdx.x * d_stride;
  const int &gi = threadIdx.y;

  opmath_t p_out_shm[d_stride] = {0.};
  opmath_t *const p_mask_shm =
      (opmath_t *)(_s) + (threadIdx.z * G + gi) * L * K;

  const scalar_t *p_offset_ptr =
      p_offset + (((bi * Q + qi) * G + gi) * L) * K * 3;
  const int mask_length = L * K;
  const int num_thread = (D / d_stride);
  const int num_iter = mask_length / num_thread;
  const int remainder = mask_length - num_iter * num_thread;
  for (int i = 0; i < num_iter; i++) {
    *(p_mask_shm + num_thread * i + threadIdx.x) =
        *(scalar_t *)(p_offset_ptr + L * K * 2 + num_thread * i + threadIdx.x);
  }
  if (remainder > 0 && threadIdx.x < remainder) {
    *(p_mask_shm + num_thread * num_iter + threadIdx.x) =
        *(scalar_t *)(p_offset_ptr + L * K * 2 + num_thread * num_iter +
                      threadIdx.x);
  }

  __syncthreads();
  // Calculate softmax over L and K
  if (threadIdx.x == 0) { // di = 0
    opmath_t softmax_max = -1e100;
    opmath_t softmax_sum = 0.0;

    // get max
    for (int j = 0; j < L * K; j++) {
      softmax_max = max(softmax_max, p_mask_shm[j]);
    }

    // get sumexp
    for (int j = 0; j < L * K; j++) {
      opmath_t exp_results = exp(p_mask_shm[j] - softmax_max);
      p_mask_shm[j] = exp_results;
      softmax_sum += exp_results;
    }

    // normalize
    for (int j = 0; j < L * K; j++) {
      p_mask_shm[j] /= softmax_sum;
    }
  }

  __syncthreads();

  int offset_idx = 0;
  int mask_idx = 0;
  const int w_stride = G * D;
  const int base_ptr = gi * D + di_s;

  for (int li = 0; li < L; li++) {
    const int spatial_h = data_spatial_shapes[li * 2];
    const int spatial_w = data_spatial_shapes[li * 2 + 1];
    const int level_start_id = data_level_start_index[li];
    const scalar_t *p_value_ptr = p_value + (bi * N + level_start_id) * G * D;

    for (int ki = 0; ki < K; ki++) {
      const opmath_t loc_w = p_offset_ptr[offset_idx];
      const opmath_t loc_h = p_offset_ptr[offset_idx + 1];
      const opmath_t attn = p_mask_shm[mask_idx];
      const opmath_t h_im = loc_h * spatial_h - 0.5;
      const opmath_t w_im = loc_w * spatial_w - 0.5;
      if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
        ms_deform_attn_im2col_bilinear<scalar_t, transfer_t, d_stride>(
            p_out_shm, p_value_ptr, spatial_h, spatial_w, h_im, w_im, attn,
            w_stride, base_ptr);
      }
      offset_idx += 2;
      mask_idx += 1;
    }
  }

  int out_idx = ((bi * Q + qi) * G + gi) * D + di_s;

  scalar_t *fp16_regs = (scalar_t *)(p_out_shm);
#pragma unroll
  for (int ds = 0; ds < d_stride; ds++) {
    fp16_regs[ds] = p_out_shm[ds];
  }

  *(transfer_t *)(p_output + out_idx) = *(transfer_t *)(p_out_shm);
}

template <typename scalar_t, int d_stride, typename transfer_t, int L, int K>
__global__ void
forward_kernel_reg(const scalar_t *p_value, const int64_t *data_spatial_shapes,
               const int64_t *data_level_start_index, const scalar_t *p_offset,
               scalar_t *p_output, const int N, const int G, const int D,
               const int Q, const int block_multiplier) {

  const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
  const int &bi = blockIdx.x * block_multiplier / Q;

  const int &di_s = threadIdx.x * d_stride;
  const int &gi = threadIdx.y;

  opmath_t p_out_shm[d_stride] = {0.};
  opmath_t p_mask_shm[L*K] = {0.};

  const scalar_t *p_offset_ptr =
      p_offset + (((bi * Q + qi) * G + gi) * L) * K * 3;

  for (int i=0; i < L*K; i++){
    p_mask_shm[i] = *(p_offset_ptr + L * K * 2 + i);
  }

  // Calculate softmax over L and K
  opmath_t softmax_max = -1e100;
  opmath_t softmax_sum = 0.0;

  // get max
  for (int j = 0; j < L * K; j++) {
    softmax_max = max(softmax_max, p_mask_shm[j]);
  }

  // get sumexp
  for (int j = 0; j < L * K; j++) {
    opmath_t exp_results = exp(p_mask_shm[j] - softmax_max);
    p_mask_shm[j] = exp_results;
    softmax_sum += exp_results;
  }

  // normalize
  for (int j = 0; j < L * K; j++) {
    p_mask_shm[j] /= softmax_sum;
  }

  int offset_idx = 0;
  int mask_idx = 0;
  const int w_stride = G * D;
  const int base_ptr = gi * D + di_s;

  for (int li = 0; li < L; li++) {
    const int spatial_h = data_spatial_shapes[li * 2];
    const int spatial_w = data_spatial_shapes[li * 2 + 1];
    const int level_start_id = data_level_start_index[li];
    const scalar_t *p_value_ptr = p_value + (bi * N + level_start_id) * G * D;

    for (int ki = 0; ki < K; ki++) {
      const opmath_t loc_w = p_offset_ptr[offset_idx];
      const opmath_t loc_h = p_offset_ptr[offset_idx + 1];
      const opmath_t attn = p_mask_shm[mask_idx];
      const opmath_t h_im = loc_h * spatial_h - 0.5;
      const opmath_t w_im = loc_w * spatial_w - 0.5;
      if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
        ms_deform_attn_im2col_bilinear<scalar_t, transfer_t, d_stride>(
            p_out_shm, p_value_ptr, spatial_h, spatial_w, h_im, w_im, attn,
            w_stride, base_ptr);
      }
      offset_idx += 2;
      mask_idx += 1;
    }
  }

  int out_idx = ((bi * Q + qi) * G + gi) * D + di_s;

  scalar_t *fp16_regs = (scalar_t *)(p_out_shm);
#pragma unroll
  for (int ds = 0; ds < d_stride; ds++) {
    fp16_regs[ds] = p_out_shm[ds];
  }

  *(transfer_t *)(p_output + out_idx) = *(transfer_t *)(p_out_shm);
}

template <typename scalar_t, typename stride_type, int K, int d_stride>
void _flash_deformable_im2col_cuda(
    cudaStream_t stream,
    const scalar_t *value,                 // B, N, G, D
    const int64_t *data_spatial_shapes,    // L * 2
    const int64_t *data_level_start_index, // L
    const scalar_t *offset,                // B, N, G, L, K, 3
    scalar_t *output,                      // B, N, G, D
    const int B, const int N, const int G, const int D, const int L,
    const int Q, const int block_thread, 
    const bool _use_reg) {

  assert(D % d_stride == 0);

  const int block_multiplier = block_thread / (D / d_stride) / G;;
  assert((B*Q) % block_multiplier == 0);
  dim3 num_blocks(B*Q / block_multiplier);
  dim3 num_threads(D / d_stride, G, block_multiplier);

  const int shm_size = 0;

  auto kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 1, K>;

  switch (L) {
  case 1:
    kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 1, K>;
    break;
  case 2:
    kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 2, K>;
    break;
  case 3:
    kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 3, K>;
    break;
  case 4:
    kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 4, K>;
    break;
  case 5:
    kernel = forward_kernel_reg<scalar_t, d_stride, stride_type, 5, K>;
    break;
  default:
    printf("L=%ld\n", L);
    throw std::invalid_argument("invalid number of scales");
  }

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shm_size);

  kernel<<<num_blocks, num_threads, shm_size, stream>>>(
      value, data_spatial_shapes, data_level_start_index, offset, output, N, G,
      D, Q, block_multiplier);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in flash_deformable_im2col_cuda: %s\n",
           cudaGetErrorString(err));
    printf("launch arguments: gridDim=(%d, %d, %d), blockDim=(%d, %d, %d), "
           "shm_size=%d, Q=%d\n\n",
           num_blocks.x, num_blocks.y, num_blocks.z, num_threads.x,
           num_threads.y, num_threads.z, shm_size, Q);
    AT_ASSERTM(false, "kernel launch error");
  }
}

template <typename scalar_t, int K>
void flash_deformable_im2col_cuda_inner(
    cudaStream_t stream,
    const scalar_t *value,                 // B, N, G, D
    const int64_t *data_spatial_shapes,    // L * 2
    const int64_t *data_level_start_index, // L
    const scalar_t *offset,                // B, N, G, L, K, 3
    scalar_t *output,                      // B, N, G, D
    const int B, const int N, const int G, const int D, const int L,
    const int Q, const int d_stride, 
    const int block_thread,
    const bool _use_reg) {
  
  assert(D % d_stride == 0);
  if(sizeof(scalar_t) == 2) {
    switch(d_stride) {
    case 1:
      _flash_deformable_im2col_cuda<scalar_t, scalar_t, K, 1>(
          stream,
          value,                  // B, N, G, D
          data_spatial_shapes,    // L * 2
          data_level_start_index, // L
          offset,                 // B, N, G, L, K, 3
          output,                 // B, N, G, D
          B, N, G, D, L, Q,
          block_thread,
          _use_reg);
      break;
    case 2:
      _flash_deformable_im2col_cuda<scalar_t, uint, K, 2>(
          stream,
          value,                  // B, N, G, D
          data_spatial_shapes,    // L * 2
          data_level_start_index, // L
          offset,                 // B, N, G, L, K, 3
          output,                 // B, N, G, D
          B, N, G, D, L, Q,
          block_thread,
          _use_reg);
      break;
    case 4:
      _flash_deformable_im2col_cuda<scalar_t, uint2, K, 4>(
          stream,
          value,                  // B, N, G, D
          data_spatial_shapes,    // L * 2
          data_level_start_index, // L
          offset,                 // B, N, G, L, K, 3
          output,                 // B, N, G, D
          B, N, G, D, L, Q,
          block_thread,
          _use_reg);
      break;
    case 8:
      _flash_deformable_im2col_cuda<scalar_t, uint4, K, 8>(
          stream,
          value,                  // B, N, G, D
          data_spatial_shapes,    // L * 2
          data_level_start_index, // L
          offset,                 // B, N, G, L, K, 3
          output,                 // B, N, G, D
          B, N, G, D, L, Q,
          block_thread,
          _use_reg);
      break;
    case 16:
      _flash_deformable_im2col_cuda<scalar_t, ulonglong4, K, 16>(
          stream,
          value,                  // B, N, G, D
          data_spatial_shapes,    // L * 2
          data_level_start_index, // L
          offset,                 // B, N, G, L, K, 3
          output,                 // B, N, G, D
          B, N, G, D, L, Q,
          block_thread,
          _use_reg);
      break;
    default:
      printf("not supported for d_stride > 16 for fp16");
      throw std::invalid_argument("invalid d_stride");
    }
  } else {
    assert(sizeof(scalar_t) == 4);
    switch(d_stride) {
    case 1:
      _flash_deformable_im2col_cuda<scalar_t, scalar_t, K, 1>(
          stream,
          value,                  // B, N, G, D
          data_spatial_shapes,    // L * 2
          data_level_start_index, // L
          offset,                 // B, N, G, L, K, 3
          output,                 // B, N, G, D
          B, N, G, D, L, Q,
          block_thread,
          _use_reg);
      break;
    case 2:
      _flash_deformable_im2col_cuda<scalar_t, uint2, K, 2>(
          stream,
          value,                  // B, N, G, D
          data_spatial_shapes,    // L * 2
          data_level_start_index, // L
          offset,                 // B, N, G, L, K, 3
          output,                 // B, N, G, D
          B, N, G, D, L, Q,
          block_thread,
          _use_reg);
      break;
    case 4:
      _flash_deformable_im2col_cuda<scalar_t, uint4, K, 4>(
          stream,
          value,                  // B, N, G, D
          data_spatial_shapes,    // L * 2
          data_level_start_index, // L
          offset,                 // B, N, G, L, K, 3
          output,                 // B, N, G, D
          B, N, G, D, L, Q,
          block_thread,
          _use_reg);
      break;
    case 8:
      _flash_deformable_im2col_cuda<scalar_t, ulonglong4, K, 8>(
          stream,
          value,                  // B, N, G, D
          data_spatial_shapes,    // L * 2
          data_level_start_index, // L
          offset,                 // B, N, G, L, K, 3
          output,                 // B, N, G, D
          B, N, G, D, L, Q,
          block_thread,
          _use_reg);
      break;
    default:
      printf("not supported for d_stride > 8 for fp32");
      throw std::invalid_argument("invalid d_stride");
    }
  }
}

template <typename scalar_t>
void flash_deformable_im2col_cuda(
    cudaStream_t stream,
    const scalar_t *value,                 // B, N, G, D
    const int64_t *data_spatial_shapes,    // L * 2
    const int64_t *data_level_start_index, // L
    const scalar_t *offset,                // B, N, G, L, K, 3
    scalar_t *output,                      // B, N, G, D
    const int B, const int N, const int G, const int D, const int L,
    const int Q, const int K, const int d_stride, 
    const int block_thread,
    const bool _use_reg) {
  switch (K) {
  case 4:
    flash_deformable_im2col_cuda_inner<scalar_t, 4>(
        stream,
        value,                  // B, N, G, D
        data_spatial_shapes,    // L * 2
        data_level_start_index, // L
        offset,                 // B, N, G, L, K, 3
        output,                 // B, N, G, D
        B, N, G, D, L, Q, d_stride, 
        block_thread, _use_reg);
    break;
  case 8:
    flash_deformable_im2col_cuda_inner<scalar_t, 8>(
        stream,
        value,                  // B, N, G, D
        data_spatial_shapes,    // L * 2
        data_level_start_index, // L
        offset,                 // B, N, G, L, K, 3
        output,                 // B, N, G, D
        B, N, G, D, L, Q, d_stride,
        block_thread, _use_reg);
    break;
  default:
    printf("not supported for K not in [4, 8]");
    throw std::invalid_argument("invalid K");
  }
}