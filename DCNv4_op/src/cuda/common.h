#ifndef FMSDACOMMON
#define FMSDACOMMON
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

#ifdef _WIN32
#define uint unsigned int
#endif

constexpr int kWarpSize = 32;
#define opmath_t at::opmath_type<scalar_t>

inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}

#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

inline bool check_backward_warpp(int d_stride, int D){
  int n_group_threads = D / d_stride;
  return (n_group_threads <= kWarpSize) && (kWarpSize % n_group_threads == 0);
}

template <typename scalar_t, typename transfer_t, int c_per_thread>
__device__ void ms_deform_attn_im2col_bilinear(
    opmath_t out_reg_array[], const scalar_t *&p_value, const int &height,
    const int &width, const opmath_t &h_px, const opmath_t &w_px,
    const opmath_t &attn, const int &w_stride, const int &base_ptr) {

  const int h_low = floor(h_px);
  const int w_low = floor(w_px);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;
  const opmath_t lh = h_px - h_low;
  const opmath_t lw = w_px - w_low;
  const opmath_t hh = 1 - lh;
  const opmath_t hw = 1 - lw;

  const opmath_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  int idx1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
  int idx2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
  int idx3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
  int idx4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;

  scalar_t v1_array[c_per_thread] = {0.};
  scalar_t v2_array[c_per_thread] = {0.};
  scalar_t v3_array[c_per_thread] = {0.};
  scalar_t v4_array[c_per_thread] = {0.};

  if (h_low >= 0 && w_low >= 0) {
    auto p1 = p_value + idx1;
    *(transfer_t *)(v1_array) = *(transfer_t *)(p1);
  }

  if (h_low >= 0 && w_high < width) {
    auto p2 = p_value + idx2;
    *(transfer_t *)(v2_array) = *(transfer_t *)(p2);
  }
  if (h_high < height && w_low >= 0) {
    auto p3 = p_value + idx3;
    *(transfer_t *)(v3_array) = *(transfer_t *)(p3);
  }
  if (h_high < height && w_high < width) {
    auto p4 = p_value + idx4;
    *(transfer_t *)(v4_array) = *(transfer_t *)(p4);
  }
#pragma unroll
  for (int i = 0; i < c_per_thread; i++) {
    out_reg_array[i] +=
        (opmath_t)attn *
        (w1 * (opmath_t)v1_array[i] + w2 * (opmath_t)v2_array[i] +
         w3 * (opmath_t)v3_array[i] + w4 * (opmath_t)v4_array[i]);
  }
}

template <typename scalar_t, typename transfer_t, int c_per_thread>
__device__ void ms_deform_attn_col2im_bilinear(
    const scalar_t *&p_value, const int &height, const int &width,
    const opmath_t &h_px, const opmath_t &w_px, const opmath_t &attn,
    const int &w_stride, const int &base_ptr, const opmath_t offset_scale_h,
    const opmath_t offset_scale_w, const scalar_t *&top_grad,
    opmath_t *&grad_im, opmath_t *grad_offset) {

  const int h_low = floor(h_px);
  const int w_low = floor(w_px);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;
  const opmath_t lh = h_px - h_low;
  const opmath_t lw = w_px - w_low;
  const opmath_t hh = 1 - lh;
  const opmath_t hw = 1 - lw;

  const opmath_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t _top_grad_array[c_per_thread] = {0.};
  *(transfer_t *)(_top_grad_array) = *(transfer_t *)(top_grad);

  opmath_t top_grad_array[c_per_thread] = {0.};
  for (int i = 0; i < c_per_thread; ++i) {
    top_grad_array[i] = (opmath_t)(_top_grad_array[i]);
  }

  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  int idx1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
  int idx2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
  int idx3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
  int idx4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;

  scalar_t v1_array[c_per_thread] = {0.};
  scalar_t v2_array[c_per_thread] = {0.};
  scalar_t v3_array[c_per_thread] = {0.};
  scalar_t v4_array[c_per_thread] = {0.};

  opmath_t grad_h_weight[c_per_thread] = {0.};
  opmath_t grad_w_weight[c_per_thread] = {0.};

  if (h_low >= 0 && w_low >= 0) {
    auto p1 = p_value + idx1;
    *(transfer_t *)(v1_array) = *(transfer_t *)(p1);
#pragma unroll
    for (int i = 0; i < c_per_thread; ++i) {
      grad_h_weight[i] -= hw * v1_array[i];
      grad_w_weight[i] -= hh * v1_array[i];
      atomicAdd(grad_im + idx1 + i, top_grad_array[i] * attn * w1);
    }
  }

  if (h_low >= 0 && w_high < width) {
    auto p2 = p_value + idx2;
    *(transfer_t *)(v2_array) = *(transfer_t *)(p2);
#pragma unroll
    for (int i = 0; i < c_per_thread; ++i) {
      grad_h_weight[i] -= lw * v2_array[i];
      grad_w_weight[i] += hh * v2_array[i];
      atomicAdd(grad_im + idx2 + i, top_grad_array[i] * attn * w2);
    }
  }
  if (h_high < height && w_low >= 0) {
    auto p3 = p_value + idx3;
    *(transfer_t *)(v3_array) = *(transfer_t *)(p3);
#pragma unroll
    for (int i = 0; i < c_per_thread; ++i) {
      grad_h_weight[i] += hw * v3_array[i];
      grad_w_weight[i] -= lh * v3_array[i];
      atomicAdd(grad_im + idx3 + i, top_grad_array[i] * attn * w3);
    }
  }
  if (h_high < height && w_high < width) {
    auto p4 = p_value + idx4;
    *(transfer_t *)(v4_array) = *(transfer_t *)(p4);
#pragma unroll
    for (int i = 0; i < c_per_thread; ++i) {
      grad_h_weight[i] += lw * v4_array[i];
      grad_w_weight[i] += lh * v4_array[i];
      atomicAdd(grad_im + idx4 + i, top_grad_array[i] * attn * w4);
    }
  }

  opmath_t _grad_offset_x = 0;
  opmath_t _grad_offset_y = 0;
#pragma unroll
  for (int i = 0; i < c_per_thread; ++i) {
    _grad_offset_x +=
        grad_w_weight[i] * top_grad_array[i]; // channel aware term
    _grad_offset_y +=
        grad_h_weight[i] * top_grad_array[i]; // channel aware term
  }
  _grad_offset_x *= (offset_scale_w * attn); // channel shared term
  _grad_offset_y *= (offset_scale_h * attn); // channel shared term

  *grad_offset = _grad_offset_x;
  *(grad_offset + 1) = _grad_offset_y;

  opmath_t current_val;
  opmath_t _grad_offset_z = 0;
#pragma unroll
  for (int i = 0; i < c_per_thread; i++) {
    current_val = (opmath_t)(w1 * v1_array[i] + w2 * v2_array[i] +
                             w3 * v3_array[i] + w4 * v4_array[i]);
    _grad_offset_z += current_val * top_grad_array[i];
  }
  *(grad_offset + 2) = _grad_offset_z;
}



#endif
