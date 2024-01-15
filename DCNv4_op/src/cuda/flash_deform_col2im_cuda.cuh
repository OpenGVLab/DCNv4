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
backward_kernel(const scalar_t *p_value, const int64_t *data_spatial_shapes,
                const int64_t *data_level_start_index, const scalar_t *p_offset,
                const scalar_t *grad_output, const int N, const int G,
                const int D, const int Q, 
                const int block_multiplier, opmath_t *grad_im,
                opmath_t *grad_offset) {

  extern __shared__ char _s[];

  const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
  const int &bi = blockIdx.x * block_multiplier / Q;

  const int &di_s = threadIdx.x * d_stride;
  const int &gi = threadIdx.y;

  opmath_t *cache_g_mask_before_softmax =
      (opmath_t *)(_s); // (block_multiplier*G) * (L * K)
  opmath_t *cache_grad_offset =
      (opmath_t *)(cache_g_mask_before_softmax +
                   block_multiplier * G * L *
                       K); // (block_multiplier*G*D/d_stride*3)
  opmath_t *const p_mask_shm =
      ((opmath_t *)(cache_grad_offset +
                    block_multiplier * G * D / d_stride * 3)) +
      (threadIdx.z * G + gi) * L * K; // G*block_multiplier * L * K

  const scalar_t *p_offset_ptr =
      p_offset + (((bi * Q + qi) * G + gi) * L) * K * 3;
  const int mask_length = L * K;
  const int num_thread = (D / d_stride);
  const int num_iter = mask_length / num_thread;
  const int remainder = mask_length - num_iter * num_thread;
  const scalar_t *top_grad = grad_output + ((bi * Q + qi) * G + gi) * D + di_s;

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
  if (threadIdx.x == 0) { // gi != 0, di = 0, li = 0
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
    opmath_t *grad_im_ptr = grad_im + (bi * N + level_start_id) * G * D;

    int cache_grad_off_idx =
        ((threadIdx.z * G + threadIdx.y) * blockDim.x + threadIdx.x) * 3;
    for (int ki = 0; ki < K; ki++) {
      const opmath_t loc_w = p_offset_ptr[offset_idx];
      const opmath_t loc_h = p_offset_ptr[offset_idx + 1];
      const opmath_t attn = p_mask_shm[mask_idx];
      const opmath_t h_im = loc_h * spatial_h - 0.5;
      const opmath_t w_im = loc_w * spatial_w - 0.5;
      // for cache_grad_offset (mG) x D/d x 3
      cache_grad_offset[cache_grad_off_idx] = 0;
      cache_grad_offset[cache_grad_off_idx + 1] = 0;
      cache_grad_offset[cache_grad_off_idx + 2] = 0;

      if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
        ms_deform_attn_col2im_bilinear<scalar_t, transfer_t, d_stride>(
            p_value_ptr, spatial_h, spatial_w, h_im, w_im, attn, w_stride,
            base_ptr, spatial_h, spatial_w, top_grad, grad_im_ptr,
            cache_grad_offset + cache_grad_off_idx);

        // aggregate across different channel for offset
        __syncthreads();
        if (threadIdx.x == 0) {
          int _didx = (threadIdx.z * G + threadIdx.y) * blockDim.x * 3;
          opmath_t _grad_w = cache_grad_offset[_didx];
          opmath_t _grad_h = cache_grad_offset[_didx + 1];
          opmath_t _grad_a = cache_grad_offset[_didx + 2];
          for (int c_id = 1; c_id < blockDim.x; ++c_id) {
            _grad_w += cache_grad_offset[_didx + 3 * c_id];
            _grad_h += cache_grad_offset[_didx + 3 * c_id + 1];
            _grad_a += cache_grad_offset[_didx + 3 * c_id + 2];
          }

          grad_offset[((bi * Q + qi) * G + gi) * L * K * 3 + li * K * 2 +
                      ki * 2] = _grad_w;
          grad_offset[((bi * Q + qi) * G + gi) * L * K * 3 + li * K * 2 +
                      ki * 2 + 1] = _grad_h;
          cache_g_mask_before_softmax
              [((threadIdx.y + threadIdx.z * G) * L + li) * K + ki] = _grad_a;
        }
      }
      __syncthreads();

      offset_idx += 2;
      mask_idx += 1;
    }
  }
  // backward for softmax
  if (threadIdx.x == 0) {
    for (int i = 0; i < L * K; ++i) {
      opmath_t grad_i = 0.;
      const opmath_t *group_g_mask = cache_g_mask_before_softmax +
                                      (threadIdx.y + threadIdx.z * G) * L * K;
      for (int j = 0; j < L * K; ++j) {
        if (i != j) {
          grad_i -= group_g_mask[j] * p_mask_shm[i] * p_mask_shm[j];
        } else {
          grad_i += group_g_mask[i] * p_mask_shm[i] * (1 - p_mask_shm[i]);
        }
      }
      grad_offset[((bi * Q + qi) * G + gi) * L * K * 3 + L * K * 2 + i] =
          grad_i;
    }
  }
  __syncthreads();
}

template <typename scalar_t, int d_stride, typename transfer_t, int L, int K>
__global__ void
backward_kernel_warp_primitive(const scalar_t *p_value, const int64_t *data_spatial_shapes,
                const int64_t *data_level_start_index, const scalar_t *p_offset,
                const scalar_t *grad_output, const int N, const int G,
                const int D, const int Q, 
                const int block_multiplier, opmath_t *grad_im,
                opmath_t *grad_offset) {

  extern __shared__ char _s[];

  const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
  const int &bi = blockIdx.x * block_multiplier / Q;

  const int &di_s = threadIdx.x * d_stride;
  const int &gi = threadIdx.y;

  const int tid = (threadIdx.z * blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;
  const int lane_id = tid % kWarpSize;
  const int group_per_warp = kWarpSize / blockDim.x;
  const int group_in_warp_id = (threadIdx.z * G + threadIdx.y) % group_per_warp;
  const unsigned lane_mask = ((1 << blockDim.x) - 1) << (group_in_warp_id * blockDim.x);

  opmath_t *cache_g_mask_before_softmax =
      (opmath_t *)(_s); // (block_multiplier*G) * (L * K)

  opmath_t *const p_mask_shm =
      ((opmath_t *)(cache_g_mask_before_softmax + block_multiplier * G * L * K)) +
        (threadIdx.z * G + gi) * L * K; // G*block_multiplier * L * K

  const scalar_t *p_offset_ptr =
      p_offset + (((bi * Q + qi) * G + gi) * L) * K * 3;
  const int mask_length = L * K;
  const int num_thread = (D / d_stride);
  const int num_iter = mask_length / num_thread;
  const int remainder = mask_length - num_iter * num_thread;
  const scalar_t *top_grad = grad_output + ((bi * Q + qi) * G + gi) * D + di_s;

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
  if (threadIdx.x == 0) { // gi != 0, di = 0, li = 0
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
    opmath_t *grad_im_ptr = grad_im + (bi * N + level_start_id) * G * D;

    int cache_grad_off_idx =
        ((threadIdx.z * G + threadIdx.y) * blockDim.x + threadIdx.x) * 3;

    opmath_t reg_grad_offset[3] = {0.};
    for (int ki = 0; ki < K; ki++) {
      const opmath_t loc_w = p_offset_ptr[offset_idx];
      const opmath_t loc_h = p_offset_ptr[offset_idx + 1];
      const opmath_t attn = p_mask_shm[mask_idx];
      const opmath_t h_im = loc_h * spatial_h - 0.5;
      const opmath_t w_im = loc_w * spatial_w - 0.5;
      reg_grad_offset[0] = 0;
      reg_grad_offset[1] = 0;
      reg_grad_offset[2] = 0;

      if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
        ms_deform_attn_col2im_bilinear<scalar_t, transfer_t, d_stride>(
            p_value_ptr, spatial_h, spatial_w, h_im, w_im, attn, w_stride,
            base_ptr, spatial_h, spatial_w, top_grad, grad_im_ptr,
            reg_grad_offset);

        // aggregate across different channel for offset
        for (uint32_t offset = blockDim.x>>1; offset > 0; offset >>= 1){
          reg_grad_offset[0] += __shfl_down_sync(lane_mask, reg_grad_offset[0], offset);
          reg_grad_offset[1] += __shfl_down_sync(lane_mask, reg_grad_offset[1], offset);
          reg_grad_offset[2] += __shfl_down_sync(lane_mask, reg_grad_offset[2], offset);
        }

        if (threadIdx.x == 0) {
          grad_offset[((bi * Q + qi) * G + gi) * L * K * 3 + li * K * 2 +
                      ki * 2] = reg_grad_offset[0];
          grad_offset[((bi * Q + qi) * G + gi) * L * K * 3 + li * K * 2 +
                      ki * 2 + 1] = reg_grad_offset[1];
          cache_g_mask_before_softmax
              [((threadIdx.y + threadIdx.z * G) * L + li) * K + ki] = reg_grad_offset[2];
        }
      }
      __syncthreads();

      offset_idx += 2;
      mask_idx += 1;
    }
  }
  // backward for softmax
  if (threadIdx.x == 0) {
    for (int i = 0; i < L * K; ++i) {
      opmath_t grad_i = 0.;
      const opmath_t *group_g_mask = cache_g_mask_before_softmax +
                                      (threadIdx.y + threadIdx.z * G) * L * K;
      for (int j = 0; j < L * K; ++j) {
        if (i != j) {
          grad_i -= group_g_mask[j] * p_mask_shm[i] * p_mask_shm[j];
        } else {
          grad_i += group_g_mask[i] * p_mask_shm[i] * (1 - p_mask_shm[i]);
        }
      }
      grad_offset[((bi * Q + qi) * G + gi) * L * K * 3 + L * K * 2 + i] =
          grad_i;
    }
  }
  __syncthreads();
}

template <typename scalar_t, typename stride_type, int K, int d_stride>
void _flash_deformable_col2im_cuda(
    cudaStream_t stream,
    const scalar_t *value,                 // B, N, G, D
    const int64_t *data_spatial_shapes,    // L * 2
    const int64_t *data_level_start_index, // L
    const scalar_t *offset,                // B, N, G, L, K, 3
    const scalar_t *grad_output,           // B, N, G, D
    const int B, const int N, const int G, const int D, const int L,
    const int Q, opmath_t *grad_im, opmath_t *grad_offset,
    const int block_thread) {

  assert(D % d_stride == 0);

  const int block_multiplier = block_thread / (D / d_stride) / G;
  assert((B*Q) % block_multiplier == 0);
  dim3 num_blocks(B*Q / block_multiplier);
  dim3 num_threads(D / d_stride, G, block_multiplier);

  int shm_size;
  if(check_backward_warpp(d_stride, D)){
    shm_size =
      sizeof(opmath_t) * (block_multiplier * G * L * K) +
      sizeof(opmath_t) * (G * block_multiplier * L * K);
  }
  else{
    shm_size =
      sizeof(opmath_t) * (block_multiplier * G * L * K) +
      sizeof(opmath_t) * (G * block_multiplier * L * K) + 
      sizeof(opmath_t) * (G * block_multiplier * D / d_stride * 3);
  }

  auto kernel = backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 1, K>;

  switch (L) {
  case 1:
    if(check_backward_warpp(d_stride, D)){
      kernel = backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 1, K>;
    } else {
      kernel = backward_kernel<scalar_t, d_stride, stride_type, 1, K>;
    }
    break;
  case 2:
    if(check_backward_warpp(d_stride, D)){
      kernel = backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 2, K>;
    } else {
      kernel = backward_kernel<scalar_t, d_stride, stride_type, 2, K>;
    }
    break;
  case 3:
    if(check_backward_warpp(d_stride, D)){
      kernel = backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 3, K>;
    } else {
      kernel = backward_kernel<scalar_t, d_stride, stride_type, 3, K>;
    }
    break;
  case 4:
    if(check_backward_warpp(d_stride, D)){
      kernel = backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 4, K>;
    } else {
      kernel = backward_kernel<scalar_t, d_stride, stride_type, 4, K>;
    }
    break;
  case 5:
    if(check_backward_warpp(d_stride, D)){
      kernel = backward_kernel_warp_primitive<scalar_t, d_stride, stride_type, 5, K>;
    } else {
      kernel = backward_kernel<scalar_t, d_stride, stride_type, 5, K>;
    }
    break;
  default:
    printf("L=%ld\n", L);
    throw std::invalid_argument("invalid number of scales");
  }
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shm_size);

  kernel<<<num_blocks, num_threads, shm_size, stream>>>(
      value, data_spatial_shapes, data_level_start_index, offset, grad_output,
      N, G, D, Q, block_multiplier, grad_im, grad_offset);

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
void flash_deformable_col2im_cuda_inner(
    cudaStream_t stream,
    const scalar_t *value,                 // B, N, G, D
    const int64_t *data_spatial_shapes,    // L * 2
    const int64_t *data_level_start_index, // L
    const scalar_t *offset,                // B, N, G, L, K, 3
    const scalar_t *grad_output,           // B, N, G, D
    const int B, const int N, const int G, const int D, const int L,
    const int Q, opmath_t *grad_im, opmath_t *grad_offset, 
    const int d_stride, const int block_thread) {

  assert(D % d_stride == 0);
  if(sizeof(scalar_t) == 2) {
    switch(d_stride) {
    case 1:
      _flash_deformable_col2im_cuda<scalar_t, scalar_t, K, 1>(
          stream,
          value,                  // B, N, G, D
          data_spatial_shapes,    // L * 2
          data_level_start_index, // L
          offset,                 // B, N, G, L, K, 3
          grad_output,            // B, N, G, D
          B, N, G, D, L, Q, grad_im, grad_offset,
          block_thread);
      break;
    case 2:
      _flash_deformable_col2im_cuda<scalar_t, uint, K, 2>(
          stream,
          value,                  // B, N, G, D
          data_spatial_shapes,    // L * 2
          data_level_start_index, // L
          offset,                 // B, N, G, L, K, 3
          grad_output,            // B, N, G, D
          B, N, G, D, L, Q, grad_im, grad_offset,
          block_thread);
      break;
    case 4:
      _flash_deformable_col2im_cuda<scalar_t, uint2, K, 4>(
          stream,
          value,                  // B, N, G, D
          data_spatial_shapes,    // L * 2
          data_level_start_index, // L
          offset,                 // B, N, G, L, K, 3
          grad_output,            // B, N, G, D
          B, N, G, D, L, Q, grad_im, grad_offset,
          block_thread);
      break;
    case 8:
      _flash_deformable_col2im_cuda<scalar_t, uint4, K, 8>(
          stream,
          value,                  // B, N, G, D
          data_spatial_shapes,    // L * 2
          data_level_start_index, // L
          offset,                 // B, N, G, L, K, 3
          grad_output,            // B, N, G, D
          B, N, G, D, L, Q, grad_im, grad_offset,
          block_thread);
      break;
    case 16:
      _flash_deformable_col2im_cuda<scalar_t, ulonglong4, K, 16>(
          stream,
          value,                  // B, N, G, D
          data_spatial_shapes,    // L * 2
          data_level_start_index, // L
          offset,                 // B, N, G, L, K, 3
          grad_output,            // B, N, G, D
          B, N, G, D, L, Q, grad_im, grad_offset,
          block_thread);
      break;
    default:
      printf("not supported for d_stride > 16 for fp16");
      throw std::invalid_argument("invalid d_stride");
    }
  } else {
    assert(sizeof(scalar_t) == 4);
    switch(d_stride) {
    case 1:  
      _flash_deformable_col2im_cuda<scalar_t, scalar_t, K, 1>(
        stream,
        value,                  // B, N, G, D
        data_spatial_shapes,    // L * 2
        data_level_start_index, // L
        offset,                 // B, N, G, L, K, 3
        grad_output,            // B, N, G, D
        B, N, G, D, L, Q, grad_im, grad_offset,
        block_thread);
      break;
    case 2:  
      _flash_deformable_col2im_cuda<scalar_t, uint2, K, 2>(
        stream,
        value,                  // B, N, G, D
        data_spatial_shapes,    // L * 2
        data_level_start_index, // L
        offset,                 // B, N, G, L, K, 3
        grad_output,            // B, N, G, D
        B, N, G, D, L, Q, grad_im, grad_offset,
        block_thread);
      break;
    case 4:  
      _flash_deformable_col2im_cuda<scalar_t, uint4, K, 4>(
        stream,
        value,                  // B, N, G, D
        data_spatial_shapes,    // L * 2
        data_level_start_index, // L
        offset,                 // B, N, G, L, K, 3
        grad_output,            // B, N, G, D
        B, N, G, D, L, Q, grad_im, grad_offset,
        block_thread);
      break;
    case 8:  
      _flash_deformable_col2im_cuda<scalar_t, ulonglong4, K, 8>(
        stream,
        value,                  // B, N, G, D
        data_spatial_shapes,    // L * 2
        data_level_start_index, // L
        offset,                 // B, N, G, L, K, 3
        grad_output,            // B, N, G, D
        B, N, G, D, L, Q, grad_im, grad_offset,
        block_thread);
      break;
    default:
      printf("not supported for d_stride > 8 for fp32");
      throw std::invalid_argument("invalid d_stride");
    }
  }
}

template <typename scalar_t>
void flash_deformable_col2im_cuda(
    cudaStream_t stream,
    const scalar_t *value,                 // B, N, G, D
    const int64_t *data_spatial_shapes,    // L * 2
    const int64_t *data_level_start_index, // L
    const scalar_t *offset,                // B, N, G, L, K, 3
    const scalar_t *grad_output,           // B, N, G, D
    const int B, const int N, const int G, const int D, const int L,
    const int Q, const int K, opmath_t *grad_im, opmath_t *grad_offset,
    const int d_stride, const int block_thread) {

  switch (K) {
  case 4:
    flash_deformable_col2im_cuda_inner<scalar_t, 4>(
        stream,
        value,                  // B, N, G, D
        data_spatial_shapes,    // L * 2
        data_level_start_index, // L
        offset,                 // B, N, G, L, K, 3
        grad_output,            // B, N, G, D
        B, N, G, D, L, Q, grad_im, grad_offset,
        d_stride, block_thread);
    break;
  case 8:
    flash_deformable_col2im_cuda_inner<scalar_t, 8>(
        stream,
        value,                  // B, N, G, D
        data_spatial_shapes,    // L * 2
        data_level_start_index, // L
        offset,                 // B, N, G, L, K, 3
        grad_output,            // B, N, G, D
        B, N, G, D, L, Q, grad_im, grad_offset,
        d_stride, block_thread);
    break;
  default:
    printf("not supported for K not in [4, 8]");
    throw std::invalid_argument("invalid K");
  }
}