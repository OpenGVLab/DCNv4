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

template <typename scalar_t, int d_stride, typename transfer_t, int L, int K,
          bool softmax>
__global__ void forward_kernel_dcn(
    const scalar_t *p_value, const scalar_t *p_offset, scalar_t *p_output,
    const int G, const int D, const int Q, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const int dilation_h, const int dilation_w,
    const int height_in, const int width_in, const int height_out,
    const int width_out, const opmath_t offset_scale, const int remove_center,
    const int block_multiplier, const int padded_offset_dim) {

  const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
  const int &bi = blockIdx.x * block_multiplier / Q;

  const int &di_s = threadIdx.x * d_stride;
  const int &gi = threadIdx.y;
  constexpr int li = 0;

  extern __shared__ char _s[];
  opmath_t *const p_mask_shm =
      (opmath_t *)(_s) + ((threadIdx.z * G + gi) * L + li) * K;

  opmath_t p_out_shm[d_stride] = {0.};

  const scalar_t *p_offset_ptr = p_offset + (bi*Q + qi)*padded_offset_dim + gi*K*3;

  const int mask_length = K;
  const int num_thread = (D / d_stride);
  const int num_iter = mask_length / num_thread;
  const int remainder = mask_length - num_iter * num_thread;

  for (int i = 0; i < num_iter; i++) {
    *(p_mask_shm + num_thread * i + threadIdx.x) =
        *(scalar_t *)(p_offset_ptr + K * 2 + num_thread * i + threadIdx.x);
  }
  if (remainder > 0 && threadIdx.x < remainder) {
    *(p_mask_shm + num_thread * num_iter + threadIdx.x) = *(
        scalar_t *)(p_offset_ptr + K * 2 + num_thread * num_iter + threadIdx.x);
  }

  int mask_idx;
  if (softmax) {
    __syncthreads();

    // Calculate softmax over L and K
    if (threadIdx.x == 0) { // gi != 0, di = 0, li = 0
      opmath_t softmax_max = -1e100;
      opmath_t softmax_sum = 0.0;

      // get max
      // #pragma unroll
      for (int j = 0; j < K; j++) {
        softmax_max = max(softmax_max, p_mask_shm[j]);
      }

      // get sumexp
      // #pragma unroll
      for (int j = 0; j < K; j++) {
        opmath_t exp_results = exp(p_mask_shm[j] - softmax_max);
        p_mask_shm[j] = exp_results;
        softmax_sum += exp_results;
      }

      // normalize
      // #pragma unroll
      for (int j = 0; j < K; j++) {
        p_mask_shm[j] /= softmax_sum;
      }
    }

    __syncthreads();
  }
  int offset_idx = 0;
  mask_idx = 0;

  const int w_stride = G * D;
  const int base_ptr = gi * D + di_s;
  const scalar_t *p_value_ptr =
      p_value + (bi * (height_in * width_in)) * (G * D);

  const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w +
                   (qi % width_out) * stride_w;
  const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h +
                   (qi / width_out) * stride_h;
  const opmath_t p0_w_ =
      p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
  const opmath_t p0_h_ =
      p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;
  const int center_h = kernel_h / 2;
  const int center_w = kernel_w / 2;

  int out_idx = ((bi * Q + qi) * G + gi) * D + di_s;

  for (int i = 0; i < kernel_w; ++i) {
    for (int j = 0; j < kernel_h; ++j) {
      if (i != center_w || j != center_h || !remove_center) {
        const opmath_t w_im =
            p0_w_ + (i * dilation_w + (opmath_t)p_offset_ptr[offset_idx]) *
                        offset_scale;
        const opmath_t h_im =
            p0_h_ + (j * dilation_h + (opmath_t)p_offset_ptr[offset_idx + 1]) *
                        offset_scale;
        const opmath_t attn = p_mask_shm[mask_idx];

        if (h_im > -1 && w_im > -1 && h_im < height_in && w_im < width_in) {
          ms_deform_attn_im2col_bilinear<scalar_t, transfer_t, d_stride>(
              p_out_shm, p_value_ptr, height_in, width_in, h_im, w_im, attn,
              w_stride, base_ptr);
        }
        offset_idx += 2;
        mask_idx += 1;
      }
    }
  }
  scalar_t *fp16_regs = (scalar_t *)(p_out_shm);
#pragma unroll
  for (int ds = 0; ds < d_stride; ds++) {
    fp16_regs[ds] = p_out_shm[ds];
  }
  *(transfer_t *)(p_output + out_idx) = *(transfer_t *)(p_out_shm);
}

template <typename scalar_t, int d_stride, typename transfer_t, int L, int K,
          bool softmax>
__global__ void forward_kernel_dcn_reg(
    const scalar_t *p_value, const scalar_t *p_offset, scalar_t *p_output,
    const int G, const int D, const int Q, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const int dilation_h, const int dilation_w,
    const int height_in, const int width_in, const int height_out,
    const int width_out, const opmath_t offset_scale, const int remove_center,
    const int block_multiplier, const int padded_offset_dim) {

  const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
  const int &bi = blockIdx.x * block_multiplier / Q;

  const int &di_s = threadIdx.x * d_stride;
  const int &gi = threadIdx.y;
  constexpr int li = 0;

  opmath_t p_mask_shm[K] = {0.};
  opmath_t p_out_shm[d_stride] = {0.};

  const scalar_t *p_offset_ptr = p_offset + (bi*Q + qi)*padded_offset_dim + gi*K*3;
  const int mask_length = K;
  const int num_thread = (D / d_stride);
  const int num_iter = mask_length / num_thread;
  const int remainder = mask_length - num_iter * num_thread;

  for (int i=0; i < K; i++){
    p_mask_shm[i] = *(p_offset_ptr + K*2 + i);
  }

  if (softmax) {
    // Calculate softmax over L and K
      opmath_t softmax_max = -1e100;
      opmath_t softmax_sum = 0.0;
      // get max
      for (int j = 0; j < K; j++) {
        softmax_max = max(softmax_max, p_mask_shm[j]);
      }

      // get sumexp
      for (int j = 0; j < K; j++) {
        opmath_t exp_results = exp(p_mask_shm[j] - softmax_max);
        p_mask_shm[j] = exp_results;
        softmax_sum += exp_results;
      }

      // normalize
      for (int j = 0; j < K; j++) {
        p_mask_shm[j] /= softmax_sum;
      }
  }

  int offset_idx = 0;
  int mask_idx = 0;

  const int w_stride = G * D;
  const int base_ptr = gi * D + di_s;
  const scalar_t *p_value_ptr =
      p_value + (bi * (height_in * width_in)) * (G * D);

  const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w +
                   (qi % width_out) * stride_w;
  const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h +
                   (qi / width_out) * stride_h;
  const opmath_t p0_w_ =
      p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
  const opmath_t p0_h_ =
      p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;
  const int center_h = kernel_h / 2;
  const int center_w = kernel_w / 2;

  int out_idx = ((bi * Q + qi) * G + gi) * D + di_s;

  for (int i = 0; i < kernel_w; ++i) {
    for (int j = 0; j < kernel_h; ++j) {
      if (i != center_w || j != center_h || !remove_center) {
        const opmath_t w_im =
            p0_w_ + (i * dilation_w + (opmath_t)p_offset_ptr[offset_idx]) *
                        offset_scale;
        const opmath_t h_im =
            p0_h_ + (j * dilation_h + (opmath_t)p_offset_ptr[offset_idx + 1]) *
                        offset_scale;
        const opmath_t attn = p_mask_shm[mask_idx];

        if (h_im > -1 && w_im > -1 && h_im < height_in && w_im < width_in) {
          ms_deform_attn_im2col_bilinear<scalar_t, transfer_t, d_stride>(
              p_out_shm, p_value_ptr, height_in, width_in, h_im, w_im, attn,
              w_stride, base_ptr);
        }
        offset_idx += 2;
        mask_idx += 1;
      }
    }
  }
  scalar_t *fp16_regs = (scalar_t *)(p_out_shm);
#pragma unroll
  for (int ds = 0; ds < d_stride; ds++) {
    fp16_regs[ds] = p_out_shm[ds];
  }

  *(transfer_t *)(p_output + out_idx) = *(transfer_t *)(p_out_shm);
}

template <typename scalar_t, typename stride_type, int d_stride>
void _dcnv4_im2col_cuda(cudaStream_t stream,
                              const scalar_t *value,    // B, H * W, (G * D)
                              const scalar_t *p_offset, // B, H * W, G * K * 3)
                              scalar_t *output,         // B, H_out*W_out, G * D
                              const int kernel_h, const int kernel_w,
                              const int stride_h, const int stride_w,
                              const int pad_h, const int pad_w,
                              const int dilation_h, const int dilation_w,
                              const int G, const int D, const int B,
                              const int height_in, const int width_in,
                              const int height_out, const int width_out,
                              const opmath_t offset_scale,
                              const int remove_center, const int block_thread,
                              const int softmax,
                              const int padded_offset_dim) {

  constexpr int L = 1;

  auto kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 9, true>;

  int N = height_in * width_in;
  int Q = height_out * width_out;
  int K = kernel_h * kernel_w;

  if (remove_center) {
    K -= 1;
  }
  if (softmax) {
    switch (K) {
    case 9:
      kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 9, true>;
      break;
    case 8:
      kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 8, true>;
      break;
    default:
      printf("K=%ld\n", K);
      throw std::invalid_argument("invalid kernel shape");
    }
  } else {
    switch (K) {
    case 9:
      kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 9, false>;
      break;
    case 8:
      kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 8, false>;
    break;
    default:
      printf("K=%ld\n", K);
      throw std::invalid_argument("invalid kernel shape");
    }
  }

  const int block_multiplier = block_thread / (D / d_stride) / G;
  assert((B*Q) % block_multiplier == 0);

  dim3 num_blocks(B*Q / block_multiplier);
  dim3 num_threads(D / d_stride, G, block_multiplier);

  int shm_size = 0;

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shm_size);

  kernel<<<num_blocks, num_threads, shm_size, stream>>>(
      value, p_offset, output, G, D, Q, kernel_h, kernel_w, stride_h, stride_w,
      pad_h, pad_w, dilation_h, dilation_w, height_in, width_in, height_out,
      width_out, offset_scale, remove_center, block_multiplier, padded_offset_dim);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in dcnv4_im2col_cuda: %s\n", cudaGetErrorString(err));
    printf("launch arguments: gridDim=(%d, %d, %d), blockDim=(%d, %d, %d), "
           "shm_size=%d\n\n",
           num_blocks.x, num_blocks.y, num_blocks.z, num_threads.x,
           num_threads.y, num_threads.z, shm_size);
    AT_ASSERTM(false, "kernel launch error");
  }
}

template <typename scalar_t>
void dcnv4_im2col_cuda(
    cudaStream_t stream,
    const scalar_t *value,    // B, H * W, (G * D)
    const scalar_t *p_offset, // B, H * W, G * K * 3)
    scalar_t *output,         // B, H_out*W_out, G * D
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int G, const int D, const int B,
    const int height_in, const int width_in, const int height_out,
    const int width_out, const opmath_t offset_scale, const int remove_center,
    const int d_stride, const int block_thread, const bool softmax,
    const int padded_offset_dim) {

  assert(D % d_stride == 0);
  if (sizeof(scalar_t) == 2) {
    switch (d_stride) {
    case 1:
      _dcnv4_im2col_cuda<scalar_t, scalar_t, 1>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 2:
      _dcnv4_im2col_cuda<scalar_t, uint, 2>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 4:
      _dcnv4_im2col_cuda<scalar_t, uint2, 4>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 8:
      _dcnv4_im2col_cuda<scalar_t, uint4, 8>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 16:
      _dcnv4_im2col_cuda<scalar_t, ulonglong4, 16>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    }
  } else {
    assert(sizeof(scalar_t) == 4);
    switch (d_stride) {
    case 1:
      _dcnv4_im2col_cuda<scalar_t, uint, 1>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 2:
      _dcnv4_im2col_cuda<scalar_t, uint2, 2>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 4:
      _dcnv4_im2col_cuda<scalar_t, uint4, 4>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 8:
      _dcnv4_im2col_cuda<scalar_t, ulonglong4, 8>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    default:
      printf("not supported for d_stride > 8 for fp32");
      throw std::invalid_argument("invalid d_stride");
    }
  }
}