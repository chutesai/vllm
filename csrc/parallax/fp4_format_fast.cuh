#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace parallax_fp4 {
__device__ __forceinline__ float decode_ue4m3(uint8_t v) {
  int e = (v >> 3) & 15, m = v & 7;
  return e ? __int_as_float((e + 120) << 23) * (1.f + (float)m * .125f)
           : (float)m * (1.f / 512.f);
}
__device__ __forceinline__ uint8_t encode_ue4m3_up(float f) {
  if (f <= 0.f) return 0;
  uint32_t bits = __float_as_uint(f);
  int e = ((bits >> 23) & 255) - 120;
  if (e < 1) {
    int m = __float2int_ru(f * 512.f);
    return m > 7 ? 8 : (uint8_t)m;
  }
  if (e > 15) return 0x7f;
  int m = ((bits & 0x7fffff) + 0xfffff) >> 20;
  if (m == 8) {
    m = 0;
    ++e;
  }
  return e > 15 ? 0x7f : (uint8_t)((e << 3) | m);
}
__device__ __forceinline__ uint8_t encode_e2m1(float q) {
  float a = fabsf(q);
  uint8_t em = a < .25f    ? 0
               : a < .75f  ? 1
               : a < 1.25f ? 2
               : a < 1.75f ? 3
               : a < 2.5f  ? 4
               : a < 3.5f  ? 5
               : a < 5.f   ? 6
                           : 7;
  return em ? (em | (q < 0.f ? 8 : 0)) : 0;
}
// Every threshold times a UE4M3 scale is exactly representable in FP32.
// The nearest FP32 input below that product is outside the quotient's
// rounding interval at the threshold, so these comparisons preserve the
// FP32-division encoder for finite inputs and positive UE4M3 scales.
__device__ __forceinline__ uint8_t encode_scaled_e2m1(float x, float scale) {
  if (scale == 0.f) return 0;
  float a = fabsf(x);
  uint8_t em = a < .25f * scale    ? 0
               : a < .75f * scale  ? 1
               : a < 1.25f * scale ? 2
               : a < 1.75f * scale ? 3
               : a < 2.5f * scale  ? 4
               : a < 3.5f * scale  ? 5
               : a < 5.f * scale   ? 6
                                   : 7;
  return em ? (em | (x < 0.f ? 8 : 0)) : 0;
}
}  // namespace parallax_fp4
