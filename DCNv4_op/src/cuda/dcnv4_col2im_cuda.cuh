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
__global__ void backward_kernel_dcn(
    const scalar_t *p_value, const scalar_t *p_offset,
    const scalar_t *grad_output, const int G, const int D, const int Q,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int height_in, const int width_in,
    const int height_out, const int width_out, const opmath_t offset_scale,
    const int remove_center, const int block_multiplier, opmath_t *grad_im,
    opmath_t *grad_offset, const int padded_offset_dim) {

  extern __shared__ char _s[];

  const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
  const int &bi = blockIdx.x * block_multiplier / Q;

  const int &di_s = threadIdx.x * d_stride;
  const int &gi = threadIdx.y;
  constexpr int li = 0;

  opmath_t *const cache_g_mask_before_softmax = (opmath_t *)(_s); // mG x K
  opmath_t *const cache_grad_offset =
      (opmath_t *)(cache_g_mask_before_softmax +
                   block_multiplier * G * K); // mG x blockDim.x x 3
  opmath_t *const p_mask_shm =
      (opmath_t *)(cache_grad_offset + block_multiplier * G * blockDim.x * 3) +
      (threadIdx.z * G + gi) * K;

  const scalar_t *p_offset_ptr = p_offset + (bi*Q + qi)*padded_offset_dim + gi*K*3;

  const int mask_length = K;
  const int num_thread = (D / d_stride);
  const int num_iter = mask_length / num_thread;
  const int remainder = mask_length - num_iter * num_thread;

  const scalar_t *top_grad = grad_output + ((bi * Q + qi) * G + gi) * D + di_s;

  __syncthreads();
  for (int i = 0; i < num_iter; i++) {
    *(p_mask_shm + num_thread * i + threadIdx.x) =
        *(scalar_t *)(p_offset_ptr + K * 2 + num_thread * i + threadIdx.x);
  }
  if (remainder > 0 && threadIdx.x < remainder) {
    *(p_mask_shm + num_thread * num_iter + threadIdx.x) = *(
        scalar_t *)(p_offset_ptr + K * 2 + num_thread * num_iter + threadIdx.x);
  }

  if (softmax) {
    __syncthreads();
    // transfer offset from global memory to shared memory >

    // Calculate softmax over L and K
    if (threadIdx.x == 0) { // gi != 0, di = 0, li = 0
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

    __syncthreads();
  }

  int offset_idx = 0;
  int mask_idx = 0;

  const int w_stride = G * D;
  const int base_ptr = gi * D + di_s;
  const scalar_t *p_value_ptr =
      p_value + (bi * (height_in * width_in)) * (G * D);
  opmath_t *grad_im_ptr = grad_im + (bi * (height_in * width_in)) * (G * D);

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

  grad_offset += (bi*Q + qi)*padded_offset_dim + gi*K*3;
  opmath_t *grad_offset_softmax = grad_offset + K * 2;

  int cache_grad_off_idx =
      ((threadIdx.z * G + threadIdx.y) * blockDim.x + threadIdx.x) * 3;
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
        cache_grad_offset[cache_grad_off_idx] = 0;
        cache_grad_offset[cache_grad_off_idx + 1] = 0;
        cache_grad_offset[cache_grad_off_idx + 2] = 0;

        if (h_im > -1 && w_im > -1 && h_im < height_in && w_im < width_in) {
          ms_deform_attn_col2im_bilinear<scalar_t, transfer_t, d_stride>(
              p_value_ptr, height_in, width_in, h_im, w_im, attn, w_stride,
              base_ptr, offset_scale, offset_scale, top_grad, grad_im_ptr,
              cache_grad_offset + cache_grad_off_idx);
        }

        // aggregated across different channel for offset
        
        __syncthreads();
        if (threadIdx.x == 0) { //
          int _didx = (threadIdx.z * G + threadIdx.y) * blockDim.x * 3;
          opmath_t _grad_w = cache_grad_offset[_didx],
                   _grad_h = cache_grad_offset[_didx + 1],
                   _grad_a = cache_grad_offset[_didx + 2];

          for (int c_id = 1; c_id < blockDim.x; ++c_id) {
            _grad_w += cache_grad_offset[_didx + 3 * c_id];
            _grad_h += cache_grad_offset[_didx + 3 * c_id + 1];
            _grad_a += cache_grad_offset[_didx + 3 * c_id + 2];
          }

          *(grad_offset) = _grad_w;     // B x H x W x G x L x K x 3
          *(grad_offset + 1) = _grad_h; // B x H x W x G x L x K x 3
          if (softmax) {
            cache_g_mask_before_softmax[(threadIdx.z * G + threadIdx.y) * K +
                                        mask_idx] = _grad_a * attn;
          }
          else{
            grad_offset_softmax[mask_idx] = _grad_a;
          }
        }
        __syncthreads();

        offset_idx += 2;
        mask_idx += 1;
        grad_offset += 2;
      }
    }
  }
  // backward for softmax
  if(softmax){
    if (threadIdx.x == 0) {
      const opmath_t* group_g_mask = cache_g_mask_before_softmax + (threadIdx.z*G + threadIdx.y)*K;
      #pragma unroll
      for (int i = 0; i < K; ++i) {
        opmath_t sum = 0.;
        for (int j = 0; j < K; ++j) {
          sum += group_g_mask[j]; // dL/di * di/dj
        }
        *(grad_offset_softmax) = group_g_mask[i] - p_mask_shm[i] * sum;

        grad_offset_softmax += 1;
      }
    }
    __syncthreads();
  }
}

template <typename scalar_t, int d_stride, typename transfer_t, int L, int K,
          bool softmax>
__global__ void backward_kernel_dcn_warp_primitive(
    const scalar_t *p_value, const scalar_t *p_offset,
    const scalar_t *grad_output, const int G, const int D, const int Q,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int height_in, const int width_in,
    const int height_out, const int width_out, const opmath_t offset_scale,
    const int remove_center, const int block_multiplier, opmath_t *grad_im,
    opmath_t *grad_offset, const int padded_offset_dim) {

  extern __shared__ char _s[];

  const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
  const int &bi = blockIdx.x * block_multiplier / Q;


  const int &di_s = threadIdx.x * d_stride;
  const int &gi = threadIdx.y;

  constexpr int li = 0;
  const int tid = (threadIdx.z * blockDim.y + threadIdx.y)*blockDim.x + threadIdx.x;

  const int lane_id = tid % kWarpSize;

  // find the position of current group in the current warp
  const int group_per_warp = kWarpSize / blockDim.x;
  const int group_in_warp_id = (threadIdx.z * G + threadIdx.y) % group_per_warp;
  const unsigned lane_mask = ((1 << blockDim.x) - 1) << (group_in_warp_id * blockDim.x);

  opmath_t *const p_mask_shm = (opmath_t *)(_s) + (threadIdx.z * G + gi) * K;
  opmath_t *cache_g_mask_before_softmax = (opmath_t *)((opmath_t *)(_s) + block_multiplier * G * K) +
                                          (threadIdx.z*G+gi)*K; // only used by threadIdx.x = 0

  const scalar_t *p_offset_ptr = p_offset + (bi*Q + qi)*padded_offset_dim + gi*K*3;

  const int mask_length = K;
  const int num_thread = (D / d_stride);
  const int num_iter = mask_length / num_thread;
  const int remainder = mask_length - num_iter * num_thread;

  const scalar_t *top_grad = grad_output + ((bi * Q + qi) * G + gi) * D + di_s;

  __syncthreads();
  for (int i = 0; i < num_iter; i++) {
    *(p_mask_shm + num_thread * i + threadIdx.x) =
        *(scalar_t *)(p_offset_ptr + K * 2 + num_thread * i + threadIdx.x);
  }
  if (remainder > 0 && threadIdx.x < remainder) {
    *(p_mask_shm + num_thread * num_iter + threadIdx.x) = *(
        scalar_t *)(p_offset_ptr + K * 2 + num_thread * num_iter + threadIdx.x);
  }

  if (softmax) {
    __syncthreads();
    // transfer offset from global memory to shared memory >

    // Calculate softmax over L and K
    if (threadIdx.x == 0) { // gi != 0, di = 0, li = 0
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

    __syncthreads();
  }

  int offset_idx = 0;
  int mask_idx = 0;

  const int w_stride = G * D;
  const int base_ptr = gi * D + di_s;
  const scalar_t *p_value_ptr =
      p_value + (bi * (height_in * width_in)) * (G * D);
  opmath_t *grad_im_ptr = grad_im + (bi * (height_in * width_in)) * (G * D);

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

  grad_offset += (bi * Q + qi)*padded_offset_dim + gi*K*3;
  opmath_t *grad_offset_softmax = grad_offset + K * 2;

  int cache_grad_off_idx =
      ((threadIdx.z * G + threadIdx.y) * blockDim.x + threadIdx.x) * 3;

  opmath_t reg_grad_offset[3] = {0.};
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
        reg_grad_offset[0] = 0;
        reg_grad_offset[1] = 0;
        reg_grad_offset[2] = 0;

        if (h_im > -1 && w_im > -1 && h_im < height_in && w_im < width_in) {
          ms_deform_attn_col2im_bilinear<scalar_t, transfer_t, d_stride>(
              p_value_ptr, height_in, width_in, h_im, w_im, attn, w_stride,
              base_ptr, offset_scale, offset_scale, top_grad, grad_im_ptr,
              reg_grad_offset);
        }

        // aggregated across different channel for offset
        for (uint32_t offset = blockDim.x>>1; offset > 0; offset >>= 1){
          reg_grad_offset[0] += __shfl_down_sync(lane_mask, reg_grad_offset[0], offset);
          reg_grad_offset[1] += __shfl_down_sync(lane_mask, reg_grad_offset[1], offset);
          reg_grad_offset[2] += __shfl_down_sync(lane_mask, reg_grad_offset[2], offset);
        }

        if (threadIdx.x == 0) { //
          *(grad_offset) = reg_grad_offset[0];     // B x H x W x G x L x K x 3
          *(grad_offset + 1) = reg_grad_offset[1]; // B x H x W x G x L x K x 3
          if (softmax) {
            cache_g_mask_before_softmax[mask_idx] = reg_grad_offset[2] * attn;
          }
          else{
            grad_offset_softmax[mask_idx] = reg_grad_offset[2];
          }
        }
        offset_idx += 2;
        mask_idx += 1;
        grad_offset += 2;
      }
    }
  }
  // backward for softmax
  if(softmax){
    if (threadIdx.x == 0) {
      opmath_t sum = 0.;
      #pragma unroll
      for (int i=0; i < K; ++i){
        sum += cache_g_mask_before_softmax[i];
      }
      #pragma unroll
      for (int i = 0; i < K; ++i) {
        *(grad_offset_softmax) = cache_g_mask_before_softmax[i] - p_mask_shm[i] * sum;
        grad_offset_softmax += 1;
      }
    }
  }
}

template <typename scalar_t, typename stride_type, int d_stride>
void _dcnv4_col2im_cuda(
    cudaStream_t stream,
    const scalar_t *value,       // B, H * W, (G * D)
    const scalar_t *p_offset,    // B, H * W, (G*K*3)
    const scalar_t *grad_output, // B, H_out*W_out, G * D
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int G, const int D, const int B,
    const int height_in, const int width_in, const int height_out,
    const int width_out, const opmath_t offset_scale, const int remove_center,
    opmath_t *grad_im, opmath_t *grad_offset, const int block_thread, 
    const bool softmax, const int padded_offset_dim) {

  constexpr int L = 1;

  auto kernel =
      backward_kernel_dcn_warp_primitive<scalar_t, d_stride, stride_type, 1, 9, false>;

  int N = height_in * width_in;
  int Q = height_out * width_out;
  int K = kernel_h * kernel_w;

  if (remove_center) {
    K -= 1;
  }

  if (softmax) {
    switch (K) {
    case 9:
      if(check_backward_warpp(d_stride, D)){
        kernel = backward_kernel_dcn_warp_primitive<scalar_t, d_stride, stride_type, 1, 9, true>;
      }
      else{
        kernel = backward_kernel_dcn<scalar_t, d_stride, stride_type, 1, 9, true>;
      }
      break;
    case 8:
      if(check_backward_warpp(d_stride, D)){
        kernel = backward_kernel_dcn_warp_primitive<scalar_t, d_stride, stride_type, 1, 8, true>;
      }
      else {
        kernel = backward_kernel_dcn<scalar_t, d_stride, stride_type, 1, 8, true>;
      }
      break;
    default:
      printf("K=%ld\n", K);
      throw std::invalid_argument("invalid kernel shape");
    }
  } else {
    switch (K) {
    case 9:
      if(check_backward_warpp(d_stride, D)){
        kernel = backward_kernel_dcn_warp_primitive<scalar_t, d_stride, stride_type, 1, 9, false>;
      }
      else{
        kernel = backward_kernel_dcn<scalar_t, d_stride, stride_type, 1, 9, false>;
      }
      break;
    case 8:
      if(check_backward_warpp(d_stride, D)){
        kernel = backward_kernel_dcn_warp_primitive<scalar_t, d_stride, stride_type, 1, 8, false>;
      }
      else {
        kernel = backward_kernel_dcn<scalar_t, d_stride, stride_type, 1, 8, false>;
      }
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

  const int blockdimX = D / d_stride;

  int shm_size = sizeof(opmath_t) * (G * block_multiplier * K) * 2;
  if(!check_backward_warpp(d_stride, D)){
    shm_size = sizeof(opmath_t) * ((G * block_multiplier * K) * 2 + G * block_multiplier * blockdimX * 3);
  }

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shm_size);

  kernel<<<num_blocks, num_threads, shm_size, stream>>>(
      value, p_offset, grad_output, G, D, Q, kernel_h, kernel_w, stride_h,
      stride_w, pad_h, pad_w, dilation_h, dilation_w, height_in, width_in,
      height_out, width_out, offset_scale, remove_center, block_multiplier,
      grad_im, grad_offset, padded_offset_dim);

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
void dcnv4_col2im_cuda(
    cudaStream_t stream,
    const scalar_t *value,       // B, H * W, (G * D)
    const scalar_t *p_offset,    // B, H * W, (G*K*3)
    const scalar_t *grad_output, // B, H_out*W_out, G * D
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int G, const int D, const int B,
    const int height_in, const int width_in, const int height_out,
    const int width_out, const opmath_t offset_scale, const int remove_center,
    opmath_t *grad_im, opmath_t *grad_offset, const int d_stride,
    const int block_thread, const bool softmax, const int padded_offset_dim) {

  assert(D % d_stride == 0);
  const int size_scalar = sizeof(scalar_t);
  if (size_scalar == 2) {
    switch (d_stride) {
    case 1:
      _dcnv4_col2im_cuda<scalar_t, scalar_t, 1>(
          stream, value, p_offset, grad_output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center, grad_im,
          grad_offset, block_thread, softmax, padded_offset_dim);
      break;
    case 2:
      _dcnv4_col2im_cuda<scalar_t, uint, 2>(
          stream, value, p_offset, grad_output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center, grad_im,
          grad_offset, block_thread, softmax, padded_offset_dim);
      break;
    case 4:
      _dcnv4_col2im_cuda<scalar_t, uint2, 4>(
          stream, value, p_offset, grad_output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center, grad_im,
          grad_offset, block_thread, softmax, padded_offset_dim);
      break;
    case 8:
      _dcnv4_col2im_cuda<scalar_t, uint4, 8>(
          stream, value, p_offset, grad_output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center, grad_im,
          grad_offset, block_thread, softmax, padded_offset_dim);
      break;
    case 16:
      _dcnv4_col2im_cuda<scalar_t, ulonglong4, 16>(
          stream, value, p_offset, grad_output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center, grad_im,
          grad_offset, block_thread, softmax, padded_offset_dim);
      break;
    }
  } else {
    assert(size_scalar == 4);
    switch (d_stride) {
    case 1:
      _dcnv4_col2im_cuda<scalar_t, uint, 1>(
          stream, value, p_offset, grad_output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center, grad_im,
          grad_offset, block_thread, softmax, padded_offset_dim);
      break;
    case 2:
      _dcnv4_col2im_cuda<scalar_t, uint2, 2>(
          stream, value, p_offset, grad_output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center, grad_im,
          grad_offset, block_thread, softmax, padded_offset_dim);
      break;
    case 4:
      _dcnv4_col2im_cuda<scalar_t, uint4, 4>(
          stream, value, p_offset, grad_output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center, grad_im,
          grad_offset, block_thread, softmax, padded_offset_dim);
      break;
    case 8:
      _dcnv4_col2im_cuda<scalar_t, ulonglong4, 8>(
          stream, value, p_offset, grad_output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center, grad_im,
          grad_offset, block_thread, softmax, padded_offset_dim);
      break;
    }
  }
}