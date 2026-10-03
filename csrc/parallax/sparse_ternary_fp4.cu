#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include "fp4_format_fast.cuh"

using namespace parallax_fp4;

__device__ __forceinline__ int scale_offset(int row, int col, int cols) {
  int cp = ((cols + 3) / 4) * 4;
  return (row / 128) * (128 * cp) + (col / 4) * 512 + (row % 32) * 16 +
         ((row % 128) / 32) * 4 + col % 4;
}

__device__ __forceinline__ float decode_fp4(unsigned char code) {
  int em = code & 7, e = em >> 1, m = em & 1;
  float mag = e ? __int_as_float((e + 126) << 23) * (1.f + .5f * m) : .5f * m;
  return (code & 8) ? -mag : mag;
}

// Candidate hardware converter. Two quotient ULPs choose the existing
// encoder's upward midpoint rule; qualify all finite BF16 values and scales.
__device__ __forceinline__ uint8_t encode_bf16_native(float x, float scale) {
  if (scale == 0.f) return 0;
  float q = fminf(__fdividef(fabsf(x), scale), 6.f);
  q = __uint_as_float(__float_as_uint(q) + 2u);
  unsigned short code;
  asm volatile(
      "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, %2, %1; mov.b16 %0, "
      "{tmp, 0}; }"
      : "=h"(code)
      : "f"(q), "f"(0.f));
  uint8_t em = code & 7;
  return em ? (em | (x < 0.f ? 8 : 0)) : 0;
}
__global__ void exhaustive_encoder_kernel(unsigned long long* counts,
                                          int* first) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 65536 * 128) return;
  int bits = index % 65536, scale_code = index / 65536;
  if ((bits & 0x7f80) == 0x7f80) return;
  float x = __uint_as_float(static_cast<unsigned>(bits) << 16);
  float scale = decode_ue4m3(static_cast<uint8_t>(scale_code));
  int expected = encode_scaled_e2m1(x, scale),
      actual = encode_bf16_native(x, scale);
  if (expected != actual) {
    atomicAdd(counts, 1ull);
    atomicMin(first, index);
  }
}
std::vector<torch::Tensor> exhaustive_encoder(torch::Tensor input) {
  const c10::cuda::CUDAGuard guard(input.device());
  auto count = torch::zeros({1}, input.options().dtype(torch::kInt64));
  auto first =
      torch::full({1}, 2147483647, input.options().dtype(torch::kInt32));
  exhaustive_encoder_kernel<<<(65536 * 128 + 255) / 256, 256, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<unsigned long long*>(count.data_ptr<int64_t>()),
      first.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {count, first};
}
template <int COMPONENTS, bool DEBUG>
__global__ void residual_quantize(const __nv_bfloat16* input, uint8_t* packed,
                                  uint8_t* scales, float* reconstruction,
                                  int rows, int k, int scale_stride) {
  int group = blockIdx.x * 8 + threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int row = group / (k / 32), col = group % (k / 32);
  if (row >= rows) return;
  float residual = __bfloat162float(input[row * k + col * 32 + lane]);
  float reconstruction_value = 0.f;
  float gain = 1.f;
#pragma unroll
  for (int component = 0; component < COMPONENTS; ++component) {
    float scaled = residual * gain;
    float maximum = fabsf(scaled);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, off));
    uint8_t scale = encode_ue4m3_up(maximum / 6.f);
    float decoded_scale = decode_ue4m3(scale);
    uint8_t value = COMPONENTS == 1 ? encode_bf16_native(scaled, decoded_scale)
                                    : encode_scaled_e2m1(scaled, decoded_scale);
    uint8_t other = __shfl_down_sync(0xffffffff, (int)value, 1);
    if ((lane & 1) == 0)
      packed[component * rows * (k / 2) + row * (k / 2) + col * 16 + lane / 2] =
          value | (other << 4);
    if (lane == 0)
      scales[component * scale_stride + scale_offset(row, col, k / 32)] = scale;
    float decoded = decode_fp4(value) * decoded_scale / gain;
    reconstruction_value += decoded;
    residual -= decoded;
    gain *= 8.f;
  }
  if constexpr (DEBUG)
    reconstruction[row * k + col * 32 + lane] = reconstruction_value;
}

std::vector<torch::Tensor> quantize(torch::Tensor input, int components,
                                    bool debug) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() &&
                  input.scalar_type() == torch::kBFloat16 && input.dim() == 2 &&
                  input.size(1) % 128 == 0,
              "Contiguous CUDA BF16 [rows,K], K%128=0 required");
  TORCH_CHECK(components >= 1 && components <= 4,
              "One to four components required");
  const c10::cuda::CUDAGuard guard(input.device());
  int rows = input.size(0), k = input.size(1);
  int ss = ((rows + 127) / 128) * 128 * ((k / 32 + 3) / 4) * 4;
  auto options = input.options().dtype(torch::kUInt8);
  auto packed = torch::empty({components, rows, k / 2}, options);
  auto scales = torch::zeros({components, ss}, options);
  auto reconstructed =
      debug ? torch::empty_like(input, input.options().dtype(torch::kFloat32))
            : torch::empty({1}, input.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
#define QUANT(C, D)                                                       \
  residual_quantize<C, D><<<(rows * (k / 32) + 7) / 8, 256, 0, stream>>>( \
      reinterpret_cast<__nv_bfloat16*>(input.data_ptr()),                 \
      packed.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),             \
      reconstructed.data_ptr<float>(), rows, k, ss)
  if (components == 1) {
    if (debug) {
      QUANT(1, true);
    } else {
      QUANT(1, false);
    }
  } else if (components == 2) {
    if (debug) {
      QUANT(2, true);
    } else {
      QUANT(2, false);
    }
  } else if (components == 3) {
    if (debug) {
      QUANT(3, true);
    } else {
      QUANT(3, false);
    }
  } else {
    if (debug) {
      QUANT(4, true);
    } else {
      QUANT(4, false);
    }
  }
#undef QUANT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales, reconstructed};
}

__global__ void pack_pair_kernel(const int8_t* codes, uint8_t* values,
                                 uint8_t* meta, int rows, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (k / 16)) return;
  int row = i / (k / 16), group = i % (k / 16);
  uint8_t metadata = 0;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    int selected[2] = {-1, -1}, count = 0;
    const int8_t* src = codes + row * k + group * 16 + half * 8;
#pragma unroll
    for (int p = 0; p < 4; ++p)
      if ((src[p * 2] || src[p * 2 + 1]) && count < 2) selected[count++] = p;
    // Input is validated for at most two active pairs before this kernel.
    for (int p = 0; count < 2 && p < 4; ++p)
      if (p != selected[0]) selected[count++] = p;
    if (selected[0] > selected[1]) {
      int t = selected[0];
      selected[0] = selected[1];
      selected[1] = t;
    }
    metadata |= (selected[0] | selected[1] << 2) << (half * 4);
    uint8_t compact = 0;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      int a = src[selected[p] * 2], b = src[selected[p] * 2 + 1];
      uint8_t av = a < 0 ? 2 : (a > 0 ? 1 : 0),
              bv = b < 0 ? 2 : (b > 0 ? 1 : 0);
      compact |= (av | (bv << 2)) << (p * 4);
    }
    values[row * (k / 8) + group * 2 + half] = compact;
  }
  meta[i] = metadata;
}

// Store each warp's four A registers and metadata contiguously. This is a
// permutation of the compact 1.5-bit representation, not another quantization.
__global__ void reorder_pair_kernel(const uint8_t* values, const uint8_t* meta,
                                    uint16_t* out, uint32_t* out_meta,
                                    int tiles, int k) {
  int tile = blockIdx.x, lane = threadIdx.x;
  if (tile >= tiles || lane >= 32) return;
  int tk = tile % (k / 128), row = (tile / (k / 128)) * 16;
  int ar = row + lane / 4, ac = (lane % 4) * 2;
  int wa = ar * (k / 8) + tk * 16 + ac;
  out[tile * 128 + lane * 4] = *reinterpret_cast<const uint16_t*>(values + wa);
  out[tile * 128 + lane * 4 + 1] =
      *reinterpret_cast<const uint16_t*>(values + wa + 8 * (k / 8));
  out[tile * 128 + lane * 4 + 2] =
      *reinterpret_cast<const uint16_t*>(values + wa + 8);
  out[tile * 128 + lane * 4 + 3] =
      *reinterpret_cast<const uint16_t*>(values + wa + 8 * (k / 8) + 8);
  int er = row + lane / 4 + (lane % 2) * 8, ec = ((lane / 2) % 2) * 4;
  out_meta[tile * 32 + lane] =
      *reinterpret_cast<const uint32_t*>(meta + er * (k / 16) + tk * 8 + ec);
}

__device__ __forceinline__ uint32_t expand_ternary(uint16_t bits) {
  uint32_t x = bits;
  x = (x | (x << 8)) & 0x00ff00ffu;
  x = (x | (x << 4)) & 0x0f0f0f0fu;
  x = (x | (x << 2)) & 0x33333333u;
  return (((x | (x >> 1)) & 0x11111111u) << 1) |
         (((x >> 1) & 0x11111111u) << 3);
}

std::vector<torch::Tensor> grouped_fp4_pack(torch::Tensor codes) {
  TORCH_CHECK(
      codes.is_cuda() && codes.is_contiguous() &&
          codes.scalar_type() == torch::kInt8 && codes.dim() == 3 &&
          codes.size(1) % 16 == 0 && codes.size(2) % 128 == 0,
      "Expected validated CUDA int8 paired sparse [E,N,K], N%16=K%128=0");
  const c10::cuda::CUDAGuard guard(codes.device());
  int rows = codes.size(0) * codes.size(1), k = codes.size(2);
  auto options = codes.options().dtype(torch::kUInt8);
  auto values = torch::empty({codes.size(0), codes.size(1), k / 8}, options);
  auto meta = torch::empty({codes.size(0), codes.size(1), k / 16}, options);
  pack_pair_kernel<<<(rows * (k / 16) + 255) / 256, 256, 0,
                     at::cuda::getCurrentCUDAStream()>>>(
      codes.data_ptr<int8_t>(), values.data_ptr<uint8_t>(),
      meta.data_ptr<uint8_t>(), rows, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto tiled_values = torch::empty_like(values),
       tiled_meta = torch::empty_like(meta);
  int tiles = (rows / 16) * (k / 128);
  reorder_pair_kernel<<<tiles, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      values.data_ptr<uint8_t>(), meta.data_ptr<uint8_t>(),
      reinterpret_cast<uint16_t*>(tiled_values.data_ptr<uint8_t>()),
      reinterpret_cast<uint32_t*>(tiled_meta.data_ptr<uint8_t>()), tiles, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {tiled_values, tiled_meta};
}

template <int COMPONENTS, int BM>
__device__ __forceinline__ void prefetch_activation(
    const uint8_t* x, const uint8_t* xs, const int* sorted, int route_base,
    int routes, int top, bool first, int input_rows, int k, int kk,
    int scale_stride, uint8_t* shared_x, uint32_t* shared_scales,
    bool sorted_hidden) {
  for (int i = threadIdx.x; i < COMPONENTS * BM * 4; i += blockDim.x) {
    int vector = i % 4, local = (i / 4) % BM, component = i / (BM * 4);
    int route = sorted[route_base + local];
    bool valid = route < routes;
    int row = valid ? (first ? route / top
                             : (sorted_hidden ? route_base + local : route))
                    : 0;
    const uint8_t* src = x + component * input_rows * (k / 2) + row * (k / 2) +
                         kk / 2 + vector * 16;
    uint8_t* dst = shared_x + (component * BM + local) * 80 + vector * 16;
    unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    int bytes = valid ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::"r"(address),
                 "l"(src), "r"(bytes));
  }
  for (int i = threadIdx.x; i < COMPONENTS * BM; i += blockDim.x) {
    int local = i % BM, component = i / BM;
    int route = sorted[route_base + local];
    bool valid = route < routes;
    int row = valid ? (first ? route / top
                             : (sorted_hidden ? route_base + local : route))
                    : 0;
    const uint8_t* src =
        xs + component * scale_stride + scale_offset(row, kk / 32, k / 32);
    unsigned address =
        static_cast<unsigned>(__cvta_generic_to_shared(shared_scales + i));
    int bytes = valid ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(address),
                 "l"(src), "r"(bytes));
  }
  asm volatile("cp.async.commit_group;");
}

template <int COMPONENTS, bool FIRST, int BM, bool QUANTIZE = false,
          bool UNGATED = false>
__global__ void multi_kernel(const uint8_t* x, const uint8_t* xs,
                             const uint8_t* weight, const uint8_t* meta,
                             const float* alpha, const float* gates,
                             const int* sorted, const int* experts,
                             const int* count, void* output, int routes,
                             int top, int n, int k, int input_rows,
                             int scale_stride, uint8_t* output_scales = nullptr,
                             int output_scale_stride = 0,
                             bool sorted_hidden = false) {
  __shared__ __nv_bfloat16
      staging[(QUANTIZE && COMPONENTS != 1) ? BM * 128 : 1];
  int route_block_index = blockIdx.y, column_block_index = blockIdx.x;
  int output_rows = sorted_hidden ? gridDim.y * BM : routes;
  if (route_block_index * BM >= *count) return;
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int expert = experts[route_block_index],
      base_n = column_block_index * 128 + warp * 16;
  if (base_n >= n) return;
  float accum[BM / 8][4] = {};
  extern __shared__ uint8_t dynamic_storage[];
  constexpr int BYTES = COMPONENTS * BM * 80;
  constexpr int STAGE_BYTES = COMPONENTS * BM * 84;
  uint8_t* first_x = dynamic_storage;
  uint32_t* first_scales = reinterpret_cast<uint32_t*>(first_x + BYTES);
  prefetch_activation<COMPONENTS, BM>(
      x, xs, sorted, route_block_index * BM, routes, top, FIRST, input_rows, k,
      0, scale_stride, first_x, first_scales, sorted_hidden);
  for (int kk = 0; kk < k; kk += 128) {
    int stage = (kk / 128) % 2;
    uint8_t* staged_x = dynamic_storage + stage * STAGE_BYTES;
    uint32_t* staged_scales = reinterpret_cast<uint32_t*>(staged_x + BYTES);
    int tile = ((expert * n + base_n) / 16) * (k / 128) + kk / 128;
    // Each lane owns a contiguous aligned 8-byte record: one vector load.
    const uint2 packed =
        reinterpret_cast<const uint2*>(weight)[tile * 32 + lane];
    uint32_t a0 = expand_ternary(static_cast<uint16_t>(packed.x));
    uint32_t a1 = expand_ternary(static_cast<uint16_t>(packed.x >> 16));
    uint32_t a2 = expand_ternary(static_cast<uint16_t>(packed.y));
    uint32_t a3 = expand_ternary(static_cast<uint16_t>(packed.y >> 16));
    uint32_t e = reinterpret_cast<const uint32_t*>(meta)[tile * 32 + lane];
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    if (kk + 128 < k) {
      uint8_t* next_x = dynamic_storage + (1 - stage) * STAGE_BYTES;
      uint32_t* next_scales = reinterpret_cast<uint32_t*>(next_x + BYTES);
      prefetch_activation<COMPONENTS, BM>(
          x, xs, sorted, route_block_index * BM, routes, top, FIRST, input_rows,
          k, kk + 128, scale_stride, next_x, next_scales, sorted_hidden);
    }
#pragma unroll
    for (int component = 0; component < COMPONENTS; ++component) {
      // UE4M3 encodings of exactly 1, 1/8, 1/64, 1/512. Applying
      // component gain in the weight scale avoids extra accumulators.
      const uint32_t gain_scales[4] = {0x38383838u, 0x20202020u, 0x08080808u,
                                       0x01010101u};
      uint32_t sa = gain_scales[component];
#pragma unroll
      for (int t = 0; t < BM / 8; ++t) {
        int local = t * 8 + lane / 4;
        const uint8_t* src =
            staged_x + (component * BM + local) * 80 + (lane % 4) * 4;
        uint32_t b0 = *reinterpret_cast<const uint32_t*>(src);
        uint32_t b1 = *reinterpret_cast<const uint32_t*>(src + 16);
        uint32_t b2 = *reinterpret_cast<const uint32_t*>(src + 32);
        uint32_t b3 = *reinterpret_cast<const uint32_t*>(src + 48);
        uint32_t sb = staged_scales[component * BM + local];
        uint16_t zero = 0;
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.sp::ordered_metadata.block_scale."
            "scale_vec::4X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3},"
            " %12, 0x0, %13, {%14,%14}, %15, {%14,%14};\n"
            : "+f"(accum[t][0]), "+f"(accum[t][1]), "+f"(accum[t][2]),
              "+f"(accum[t][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
              "r"(b3), "r"(e), "r"(sa), "h"(zero), "r"(sb));
      }
    }
    __syncthreads();
  }
  if constexpr (QUANTIZE && COMPONENTS == 1) {
    __shared__ float maxima[BM * 8];
#pragma unroll
    for (int t = 0; t < BM / 8; ++t) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int col = base_n + lane / 4 + (j / 2) * 8;
        float value = accum[t][j] * alpha[expert * n + col];
        value = __bfloat162float(__float2bfloat16_rn(value));
        value = fmaxf(value, 0.f);
        value *= value;
        accum[t][j] = __bfloat162float(__float2bfloat16_rn(value));
      }
      float even = fmaxf(fabsf(accum[t][0]), fabsf(accum[t][2]));
      float odd = fmaxf(fabsf(accum[t][1]), fabsf(accum[t][3]));
#pragma unroll
      for (int offset = 4; offset <= 16; offset *= 2) {
        even = fmaxf(even, __shfl_xor_sync(0xffffffff, even, offset));
        odd = fmaxf(odd, __shfl_xor_sync(0xffffffff, odd, offset));
      }
      if (lane / 4 == 0) {
        int local = t * 8 + (lane % 4) * 2;
        maxima[local * 8 + warp] = even;
        maxima[(local + 1) * 8 + warp] = odd;
      }
    }
    __syncthreads();
#pragma unroll
    for (int t = 0; t < BM / 8; ++t) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int local = t * 8 + (lane % 4) * 2 + (j % 2);
        int route = sorted[route_block_index * BM + local];
        int col = base_n + lane / 4 + (j / 2) * 8;
        float maximum =
            fmaxf(maxima[local * 8 + warp], maxima[local * 8 + (warp ^ 1)]);
        uint8_t scale = encode_ue4m3_up(maximum / 6.f);
        uint8_t value = encode_bf16_native(accum[t][j], decode_ue4m3(scale));
        uint8_t other = __shfl_xor_sync(0xffffffff, (int)value, 4);
        if (route < routes && (lane / 4) % 2 == 0) {
          int destination =
              sorted_hidden ? route_block_index * BM + local : route;
          reinterpret_cast<uint8_t*>(output)[destination * (n / 2) + col / 2] =
              value | (other << 4);
          if (col % 32 == 0)
            output_scales[scale_offset(destination, col / 32, n / 32)] = scale;
        }
      }
    }
  } else {
#pragma unroll
    for (int t = 0; t < BM / 8; ++t) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int route =
            sorted[route_block_index * BM + t * 8 + (lane % 4) * 2 + (j % 2)];
        int col = base_n + lane / 4 + (j / 2) * 8;
        if (route < routes && col < n) {
          float value = accum[t][j] * alpha[expert * n + col];
          value = __bfloat162float(__float2bfloat16_rn(value));
          if constexpr (FIRST) {
            value = fmaxf(value, 0.f);
            value *= value;
            if constexpr (QUANTIZE) {
              int local = t * 8 + (lane % 4) * 2 + (j % 2);
              staging[local * 128 + warp * 16 + lane / 4 + (j / 2) * 8] =
                  __float2bfloat16_rn(value);
            } else
              reinterpret_cast<__nv_bfloat16*>(output)[route * n + col] =
                  __float2bfloat16_rn(value);
          } else {
            float contribution = value * gates[route];
            if constexpr (UNGATED)
              reinterpret_cast<__nv_bfloat16*>(output)[route * n + col] =
                  __float2bfloat16_rn(value);
            else
              reinterpret_cast<float*>(output)[route * n + col] = contribution;
          }
        }
      }
    }
    if constexpr (QUANTIZE) {
      __syncthreads();
      for (int group = warp; group < BM * 4; group += 8) {
        int local = group / 4, half = group % 4;
        int route = sorted[route_block_index * BM + local];
        if (route < routes) {
          float residual =
              __bfloat162float(staging[local * 128 + half * 32 + lane]);
          float gain = 1.f;
#pragma unroll
          for (int component = 0; component < COMPONENTS; ++component) {
            float scaled = residual * gain, maximum = fabsf(scaled);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
              maximum =
                  fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, off));
            uint8_t scale = encode_ue4m3_up(maximum / 6.f);
            float decoded_scale = decode_ue4m3(scale);
            uint8_t value = encode_scaled_e2m1(scaled, decoded_scale);
            uint8_t other = __shfl_down_sync(0xffffffff, (int)value, 1);
            if ((lane & 1) == 0)
              reinterpret_cast<uint8_t*>(
                  output)[component * output_rows * (n / 2) +
                          (sorted_hidden ? route_block_index * BM + local
                                         : route) *
                              (n / 2) +
                          column_block_index * 64 + half * 16 + lane / 2] =
                  value | (other << 4);
            if (lane == 0)
              output_scales[component * output_scale_stride +
                            scale_offset(
                                sorted_hidden ? route_block_index * BM + local
                                              : route,
                                column_block_index * 4 + half, n / 32)] = scale;
            residual -= decode_fp4(value) * decoded_scale / gain;
            gain *= 8.f;
          }
        }
      }
    }
  }
}

// Two 16-column weight fragments per warp reuse every activation fragment.
// A warp owns all 32 columns of a quantization group, so its scale reduction
// needs no exchange with another warp.
template <bool FIRST, int BM, bool QUANTIZE = false, bool UNGATED = false>
__global__ void dual_kernel(const uint8_t* x, const uint8_t* xs,
                            const uint8_t* weight, const uint8_t* meta,
                            const float* alpha, const float* gates,
                            const int* sorted, const int* experts,
                            const int* count, void* output, int routes, int top,
                            int n, int k, int input_rows, int scale_stride,
                            uint8_t* output_scales = nullptr,
                            int output_scale_stride = 0,
                            bool sorted_hidden = false) {
  int route_block_index = blockIdx.y, column_block_index = blockIdx.x;
  int output_rows = sorted_hidden ? gridDim.y * BM : routes;
  if (route_block_index * BM >= *count) return;
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int expert = experts[route_block_index],
      base_n = column_block_index * 128 + warp * 32;
  float accum[2][BM / 8][4] = {};
  extern __shared__ uint8_t storage[];
  constexpr int BYTES = BM * 80, STAGE_BYTES = BM * 84;
  prefetch_activation<1, BM>(x, xs, sorted, route_block_index * BM, routes, top,
                             FIRST, input_rows, k, 0, scale_stride, storage,
                             reinterpret_cast<uint32_t*>(storage + BYTES),
                             sorted_hidden);
  for (int kk = 0; kk < k; kk += 128) {
    int stage = (kk / 128) % 2;
    uint8_t* staged_x = storage + stage * STAGE_BYTES;
    uint32_t* staged_scales = reinterpret_cast<uint32_t*>(staged_x + BYTES);
    uint32_t a[2][4], e[2];
#pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
      int tile =
          ((expert * n + base_n + chunk * 16) / 16) * (k / 128) + kk / 128;
      uint2 packed = reinterpret_cast<const uint2*>(weight)[tile * 32 + lane];
      a[chunk][0] = expand_ternary(static_cast<uint16_t>(packed.x));
      a[chunk][1] = expand_ternary(static_cast<uint16_t>(packed.x >> 16));
      a[chunk][2] = expand_ternary(static_cast<uint16_t>(packed.y));
      a[chunk][3] = expand_ternary(static_cast<uint16_t>(packed.y >> 16));
      e[chunk] = reinterpret_cast<const uint32_t*>(meta)[tile * 32 + lane];
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    if (kk + 128 < k) {
      uint8_t* next = storage + (1 - stage) * STAGE_BYTES;
      prefetch_activation<1, BM>(
          x, xs, sorted, route_block_index * BM, routes, top, FIRST, input_rows,
          k, kk + 128, scale_stride, next,
          reinterpret_cast<uint32_t*>(next + BYTES), sorted_hidden);
    }
#pragma unroll
    for (int t = 0; t < BM / 8; ++t) {
      int local = t * 8 + lane / 4;
      const uint8_t* src = staged_x + local * 80 + (lane % 4) * 4;
      uint32_t b0 = *reinterpret_cast<const uint32_t*>(src);
      uint32_t b1 = *reinterpret_cast<const uint32_t*>(src + 16);
      uint32_t b2 = *reinterpret_cast<const uint32_t*>(src + 32);
      uint32_t b3 = *reinterpret_cast<const uint32_t*>(src + 48);
      uint32_t sb = staged_scales[local];
      uint16_t zero = 0;
#pragma unroll
      for (int chunk = 0; chunk < 2; ++chunk) {
        uint32_t sa = 0x38383838u;
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.sp::ordered_metadata.block_scale."
            "scale_vec::4X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3},"
            " %12, 0x0, %13, {%14,%14}, %15, {%14,%14};\n"
            : "+f"(accum[chunk][t][0]), "+f"(accum[chunk][t][1]),
              "+f"(accum[chunk][t][2]), "+f"(accum[chunk][t][3])
            : "r"(a[chunk][0]), "r"(a[chunk][1]), "r"(a[chunk][2]),
              "r"(a[chunk][3]), "r"(b0), "r"(b1), "r"(b2), "r"(b3),
              "r"(e[chunk]), "r"(sa), "h"(zero), "r"(sb));
      }
    }
    __syncthreads();
  }
  if constexpr (QUANTIZE) {
#pragma unroll
    for (int t = 0; t < BM / 8; ++t) {
#pragma unroll
      for (int chunk = 0; chunk < 2; ++chunk) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          int col = base_n + chunk * 16 + lane / 4 + (j / 2) * 8;
          float value = accum[chunk][t][j] * alpha[expert * n + col];
          value = __bfloat162float(__float2bfloat16_rn(value));
          value = fmaxf(value, 0.f);
          value *= value;
          accum[chunk][t][j] = __bfloat162float(__float2bfloat16_rn(value));
        }
      }
      float even = fmaxf(fmaxf(fabsf(accum[0][t][0]), fabsf(accum[0][t][2])),
                         fmaxf(fabsf(accum[1][t][0]), fabsf(accum[1][t][2])));
      float odd = fmaxf(fmaxf(fabsf(accum[0][t][1]), fabsf(accum[0][t][3])),
                        fmaxf(fabsf(accum[1][t][1]), fabsf(accum[1][t][3])));
#pragma unroll
      for (int offset = 4; offset <= 16; offset *= 2) {
        even = fmaxf(even, __shfl_xor_sync(0xffffffff, even, offset));
        odd = fmaxf(odd, __shfl_xor_sync(0xffffffff, odd, offset));
      }
#pragma unroll
      for (int chunk = 0; chunk < 2; ++chunk) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          int local = t * 8 + (lane % 4) * 2 + j % 2;
          int route = sorted[route_block_index * BM + local];
          int col = base_n + chunk * 16 + lane / 4 + (j / 2) * 8;
          uint8_t scale = encode_ue4m3_up((j % 2 ? odd : even) / 6.f);
          uint8_t value =
              encode_bf16_native(accum[chunk][t][j], decode_ue4m3(scale));
          uint8_t other = __shfl_xor_sync(0xffffffff, (int)value, 4);
          if (route < routes && (lane / 4) % 2 == 0) {
            int dest = sorted_hidden ? route_block_index * BM + local : route;
            reinterpret_cast<uint8_t*>(output)[dest * (n / 2) + col / 2] =
                value | (other << 4);
            if (col % 32 == 0)
              output_scales[scale_offset(dest, col / 32, n / 32)] = scale;
          }
        }
      }
    }
  } else {
#pragma unroll
    for (int chunk = 0; chunk < 2; ++chunk) {
#pragma unroll
      for (int t = 0; t < BM / 8; ++t) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          int route =
              sorted[route_block_index * BM + t * 8 + (lane % 4) * 2 + j % 2];
          int col = base_n + chunk * 16 + lane / 4 + (j / 2) * 8;
          if (route < routes) {
            float value = accum[chunk][t][j] * alpha[expert * n + col];
            value = __bfloat162float(__float2bfloat16_rn(value));
            if constexpr (FIRST) {
              value = fmaxf(value, 0.f);
              value *= value;
            }
            if constexpr (FIRST || UNGATED)
              reinterpret_cast<__nv_bfloat16*>(output)[route * n + col] =
                  __float2bfloat16_rn(value);
            else
              reinterpret_cast<float*>(output)[route * n + col] =
                  value * gates[route];
          }
        }
      }
    }
  }
}

torch::Tensor run(torch::Tensor x, torch::Tensor scales, torch::Tensor weight,
                  torch::Tensor meta, torch::Tensor alpha, torch::Tensor gates,
                  torch::Tensor sorted, torch::Tensor experts,
                  torch::Tensor count, int64_t routes, int64_t top, bool first,
                  int64_t bm, bool ungated_reduce, bool sorted_hidden = false) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                  x.scalar_type() == torch::kUInt8 && x.dim() == 3,
              "Packed CUDA components required");
  const c10::cuda::CUDAGuard guard(x.device());
  int components = x.size(0), rows = x.size(1), k = x.size(2) * 2,
      n = weight.size(1);
  TORCH_CHECK((components >= 1 && components <= 4) && k % 128 == 0 &&
                  n % 128 == 0 && weight.size(2) * 8 == k,
              "Unsupported FP4 projection shape");
  TORCH_CHECK(alpha.is_contiguous() && alpha.scalar_type() == torch::kFloat32 &&
                  alpha.dim() == 2 && alpha.size(0) == weight.size(0) &&
                  alpha.size(1) == n,
              "FP32 scales per expert/output row required");
  TORCH_CHECK((bm == 16 || bm == 32 || bm == 64 || bm == 128) &&
                  sorted.numel() % bm == 0,
              "Supported routing tile required");
  TORCH_CHECK(scales.dim() == 2 && scales.size(0) == components &&
                  scales.is_contiguous(),
              "Matching component scales required");
  TORCH_CHECK(!ungated_reduce || (!first && top > 0 && routes % top == 0),
              "Ungated reduction requires valid down routes");
  auto out =
      torch::empty({routes, n}, x.options().dtype((first || ungated_reduce)
                                                      ? torch::kBFloat16
                                                      : torch::kFloat32));
  dim3 grid((n + 127) / 128, sorted.numel() / bm);
  auto stream = at::cuda::getCurrentCUDAStream();
#define RUN(C, F, B)                                                         \
  if constexpr (C == 1) {                                                    \
    if (ungated_reduce) {                                                    \
      dual_kernel<false, B, false, true><<<grid, 128, 2 * B * 84, stream>>>( \
          x.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),                 \
          weight.data_ptr<uint8_t>(), meta.data_ptr<uint8_t>(),              \
          alpha.data_ptr<float>(), gates.data_ptr<float>(),                  \
          sorted.data_ptr<int>(), experts.data_ptr<int>(),                   \
          count.data_ptr<int>(), out.data_ptr(), routes, top, n, k, rows,    \
          scales.size(1), nullptr, 0, sorted_hidden);                        \
    } else {                                                                 \
      dual_kernel<F, B><<<grid, 128, 2 * B * 84, stream>>>(                  \
          x.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),                 \
          weight.data_ptr<uint8_t>(), meta.data_ptr<uint8_t>(),              \
          alpha.data_ptr<float>(), gates.data_ptr<float>(),                  \
          sorted.data_ptr<int>(), experts.data_ptr<int>(),                   \
          count.data_ptr<int>(), out.data_ptr(), routes, top, n, k, rows,    \
          scales.size(1), nullptr, 0, sorted_hidden);                        \
    }                                                                        \
  } else if (ungated_reduce) {                                               \
    C10_CUDA_CHECK(cudaFuncSetAttribute(                                     \
        multi_kernel<C, false, B, false, true>,                              \
        cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));                \
    multi_kernel<C, false, B, false, true>                                   \
        <<<grid, 256, 2 * C * B * 84, stream>>>(                             \
            x.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),               \
            weight.data_ptr<uint8_t>(), meta.data_ptr<uint8_t>(),            \
            alpha.data_ptr<float>(), gates.data_ptr<float>(),                \
            sorted.data_ptr<int>(), experts.data_ptr<int>(),                 \
            count.data_ptr<int>(), out.data_ptr(), routes, top, n, k, rows,  \
            scales.size(1), nullptr, 0, sorted_hidden);                      \
  } else {                                                                   \
    C10_CUDA_CHECK(cudaFuncSetAttribute(                                     \
        multi_kernel<C, F, B>, cudaFuncAttributeMaxDynamicSharedMemorySize,  \
        65536));                                                             \
    multi_kernel<C, F, B><<<grid, 256, 2 * C * B * 84, stream>>>(            \
        x.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),                   \
        weight.data_ptr<uint8_t>(), meta.data_ptr<uint8_t>(),                \
        alpha.data_ptr<float>(), gates.data_ptr<float>(),                    \
        sorted.data_ptr<int>(), experts.data_ptr<int>(),                     \
        count.data_ptr<int>(), out.data_ptr(), routes, top, n, k, rows,      \
        scales.size(1), nullptr, 0, sorted_hidden);                          \
  }
#define PICK_B(C, F)     \
  if (bm == 16) {        \
    RUN(C, F, 16);       \
  } else if (bm == 32) { \
    RUN(C, F, 32);       \
  } else if (bm == 64) { \
    RUN(C, F, 64);       \
  } else {               \
    RUN(C, F, 128);      \
  }
  if (components == 1) {
    if (first) {
      PICK_B(1, true)
    } else {
      PICK_B(1, false)
    }
  } else if (components == 2) {
    if (first) {
      PICK_B(2, true)
    } else {
      PICK_B(2, false)
    }
  } else if (components == 3) {
    if (first) {
      PICK_B(3, true)
    } else {
      PICK_B(3, false)
    }
  } else {
    if (first) {
      PICK_B(4, true)
    } else {
      PICK_B(4, false)
    }
  }
#undef PICK_B
#undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> run_up_quantized(
    torch::Tensor x, torch::Tensor scales, torch::Tensor weight,
    torch::Tensor meta, torch::Tensor alpha, torch::Tensor sorted,
    torch::Tensor experts, torch::Tensor count, int64_t routes, int64_t top,
    int64_t bm, bool sorted_hidden = false) {
  const c10::cuda::CUDAGuard guard(x.device());
  int components = x.size(0), rows = x.size(1), k = x.size(2) * 2,
      n = weight.size(1);
  TORCH_CHECK((components >= 1 && components <= 4) && k % 128 == 0 &&
                  n % 128 == 0 && weight.size(2) * 8 == k,
              "Unsupported FP4 projection shape");
  TORCH_CHECK((bm == 16 || bm == 32 || bm == 64 || bm == 128) &&
                  sorted.numel() % bm == 0,
              "Unsupported route tile");
  TORCH_CHECK(alpha.scalar_type() == torch::kFloat32 && alpha.dim() == 2 &&
                  alpha.size(0) == weight.size(0) && alpha.size(1) == n,
              "Per-output scales required");
  int output_rows = sorted_hidden ? sorted.numel() : routes;
  int ss = ((output_rows + 127) / 128) * 128 * ((n / 32 + 3) / 4) * 4;
  auto packed = torch::empty({components, output_rows, n / 2}, x.options());
  auto output_scales = torch::zeros({components, ss}, x.options());
  dim3 grid((n + 127) / 128, sorted.numel() / bm);
  auto stream = at::cuda::getCurrentCUDAStream();
#define UP(C, B)                                                           \
  if constexpr (C == 1) {                                                  \
    dual_kernel<true, B, true><<<grid, 128, 2 * B * 84, stream>>>(         \
        x.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),                 \
        weight.data_ptr<uint8_t>(), meta.data_ptr<uint8_t>(),              \
        alpha.data_ptr<float>(), nullptr, sorted.data_ptr<int>(),          \
        experts.data_ptr<int>(), count.data_ptr<int>(), packed.data_ptr(), \
        routes, top, n, k, rows, scales.size(1),                           \
        output_scales.data_ptr<uint8_t>(), ss, sorted_hidden);             \
  } else {                                                                 \
    C10_CUDA_CHECK(cudaFuncSetAttribute(                                   \
        multi_kernel<C, true, B, true>,                                    \
        cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));              \
    multi_kernel<C, true, B, true><<<grid, 256, 2 * C * B * 84, stream>>>( \
        x.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),                 \
        weight.data_ptr<uint8_t>(), meta.data_ptr<uint8_t>(),              \
        alpha.data_ptr<float>(), nullptr, sorted.data_ptr<int>(),          \
        experts.data_ptr<int>(), count.data_ptr<int>(), packed.data_ptr(), \
        routes, top, n, k, rows, scales.size(1),                           \
        output_scales.data_ptr<uint8_t>(), ss, sorted_hidden);             \
  }
#define PICK(C)          \
  if (bm == 16) {        \
    UP(C, 16);           \
  } else if (bm == 32) { \
    UP(C, 32);           \
  } else if (bm == 64) { \
    UP(C, 64);           \
  } else {               \
    UP(C, 128);          \
  }
  if (components == 1) {
    PICK(1)
  } else if (components == 2) {
    PICK(2)
  } else if (components == 3) {
    PICK(3)
  } else {
    PICK(4)
  }
#undef PICK
#undef UP
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, output_scales};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("exhaustive_encoder", &exhaustive_encoder);
  m.def("quantize", &quantize);
  m.def("pack", &grouped_fp4_pack);
  m.def("run", &run);
  m.def("run_up_quantized", &run_up_quantized);
}
