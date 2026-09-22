/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_S_W_OFF 0
#define SMEM_S_W_STAGE_BYTES 16384
#define SMEM_S_W_STRIDE 40960
#define SMEM_S_KN_OFF 16384
#define SMEM_S_KN_STAGE_BYTES 16384
#define SMEM_S_KN_STRIDE 40960
#define SMEM_S_U_OFF 32768
#define SMEM_S_U_STAGE_BYTES 8192
#define SMEM_S_U_STRIDE 40960
#define SMEM_S_H_OFF 81920
#define SMEM_S_H_STAGE_BYTES 16384
#define SMEM_S_H_STRIDE 16384
#define SMEM_S_VDEC_OFF 98304
#define SMEM_S_VDEC_STAGE_BYTES 8192
#define SMEM_S_VDEC_STRIDE 8192
#define SMEM_S_G_OFF 106496
#define SMEM_S_G_STAGE_BYTES 256
#define SMEM_S_G_STRIDE 256
#define SMEM_TOTAL 106752
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void fma_f32x2_noftz_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("add.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("sub.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
    float2 r;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ void fma_scale_x32(
    float* sv, const float2* scale2, const float2* neg_max2)
{
    float2* sv_2 = reinterpret_cast<float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++)
        fma_f32x2_inplace(&sv_2[j], *scale2, *neg_max2);
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 add_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rn_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rz_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rz_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rz.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rm_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rm.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rm_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rm.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rp_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rp.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rp_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rp.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rn_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rz_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rz_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rz.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rm_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rm.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rm_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rm.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rp_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rp.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rp_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rp.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rz_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rz_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rz_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rz.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rz_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rz.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rm_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rm.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rm_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rm.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rm_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rm.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rm_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rm.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rp_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rp.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rp_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rp.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rp_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rp.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rp_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rp.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_gdn_chunk_train_d0a2cae4e982e3c58b06(__nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ u, float* __restrict__ g_cs, float* __restrict__ h0, __nv_bfloat16* __restrict__ h_out, __nv_bfloat16* __restrict__ v_new, float* __restrict__ final_state, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int* __restrict__ seq_chunk_start, int num_heads, int num_v_heads, int use_initial_state, int store_final_state)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* s_w = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_w_addr = smem + 0;
    __nv_bfloat16* s_kn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_kn_addr = smem + 16384;
    __nv_bfloat16* s_u = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_u_addr = smem + 32768;
    __nv_bfloat16* s_h = reinterpret_cast<__nv_bfloat16*>(smem_raw + 81920);
    const int s_h_addr = smem + 81920;
    __nv_bfloat16* s_vdec = reinterpret_cast<__nv_bfloat16*>(smem_raw + 98304);
    const int s_vdec_addr = smem + 98304;
    float* s_g = reinterpret_cast<float*>(smem_raw + 106496);
    const int s_g_addr = smem + 106496;

    // === Task calls (dependency order) ===
    int seq = blockIdx.x;
    int hv = blockIdx.y;
    int v0 = blockIdx.z * 16;
    int c_begin = seq_chunk_start[seq];
    int c_end = seq_chunk_start[seq + 1];
    int group = num_v_heads / num_heads;
    int hq = hv / group;
    int lane_0 = lane;
    int warp_1 = warp;
    int tid_2 = tid;
    int col_q = lane_0 % 4 * 2;
    int k_row0 = warp_1 * 32 + lane_0 / 4;
    int tok_row_a = warp_1 * 16 + lane_0 / 4;
    int tok_row_b = tok_row_a + 8;
    float acc_h[16];
    float acc_v[8];
    unsigned int a_frag_k[4];
    unsigned int a_frag_mn[4];
    unsigned int b_frag[4];
    long long state_base = ((long long)seq * (long long)num_v_heads + (long long)hv) * 16384;
    acc_h[0] = 0.0f;
    acc_h[1] = 0.0f;
    acc_h[2] = 0.0f;
    acc_h[3] = 0.0f;
    acc_h[4] = 0.0f;
    acc_h[5] = 0.0f;
    acc_h[6] = 0.0f;
    acc_h[7] = 0.0f;
    acc_h[8] = 0.0f;
    acc_h[9] = 0.0f;
    acc_h[10] = 0.0f;
    acc_h[11] = 0.0f;
    acc_h[12] = 0.0f;
    acc_h[13] = 0.0f;
    acc_h[14] = 0.0f;
    acc_h[15] = 0.0f;
    if (use_initial_state != 0) {
        #pragma unroll
        for (int mb = 0; mb < 2; mb++) {
            #pragma unroll
            for (int nb = 0; nb < 2; nb++) {
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    int krow = k_row0 + mb * 16 + half * 8;
                    {
                        float2 _v2_0 = *reinterpret_cast<const float2*>(h0 + state_base + (long long)krow * 128 + (long long)v0 + (long long)(nb * 8) + (long long)col_q);
                        acc_h[mb * 2 * 4 + nb * 4 + half * 2] = _v2_0.x;
                        acc_h[mb * 2 * 4 + nb * 4 + half * 2 + 1] = _v2_0.y;
                    }
                }
            }
        }
    }
    int num_chunks = c_end - c_begin;
    int stage_off = 0;
    int stage_tok0 = chunk_start[c_begin];
    int stage_len = chunk_len[c_begin];
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_w_addr + (unsigned int)stage_off + (unsigned int)(tid % 16 * 8 / 64 * 8192 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(w + (((long long)(stage_tok0 + tid / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)(tid % 16 * 8))), "r"((stage_len > tid / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_w_addr + (unsigned int)stage_off + (unsigned int)((128 + tid) % 16 * 8 / 64 * 8192 + ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 ^ ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(w + (((long long)(stage_tok0 + (128 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((128 + tid) % 16 * 8))), "r"((stage_len > (128 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_w_addr + (unsigned int)stage_off + (unsigned int)((256 + tid) % 16 * 8 / 64 * 8192 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(w + (((long long)(stage_tok0 + (256 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((256 + tid) % 16 * 8))), "r"((stage_len > (256 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_w_addr + (unsigned int)stage_off + (unsigned int)((384 + tid) % 16 * 8 / 64 * 8192 + ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 ^ ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(w + (((long long)(stage_tok0 + (384 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((384 + tid) % 16 * 8))), "r"((stage_len > (384 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_w_addr + (unsigned int)stage_off + (unsigned int)((512 + tid) % 16 * 8 / 64 * 8192 + ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 ^ ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(w + (((long long)(stage_tok0 + (512 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((512 + tid) % 16 * 8))), "r"((stage_len > (512 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_w_addr + (unsigned int)stage_off + (unsigned int)((640 + tid) % 16 * 8 / 64 * 8192 + ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 ^ ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(w + (((long long)(stage_tok0 + (640 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((640 + tid) % 16 * 8))), "r"((stage_len > (640 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_w_addr + (unsigned int)stage_off + (unsigned int)((768 + tid) % 16 * 8 / 64 * 8192 + ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 ^ ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(w + (((long long)(stage_tok0 + (768 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((768 + tid) % 16 * 8))), "r"((stage_len > (768 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_w_addr + (unsigned int)stage_off + (unsigned int)((896 + tid) % 16 * 8 / 64 * 8192 + ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 ^ ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(w + (((long long)(stage_tok0 + (896 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((896 + tid) % 16 * 8))), "r"((stage_len > (896 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)stage_off + (unsigned int)(tid % 16 * 8 / 64 * 8192 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(stage_tok0 + tid / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)(tid % 16 * 8))), "r"((stage_len > tid / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)stage_off + (unsigned int)((128 + tid) % 16 * 8 / 64 * 8192 + ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 ^ ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(stage_tok0 + (128 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((128 + tid) % 16 * 8))), "r"((stage_len > (128 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)stage_off + (unsigned int)((256 + tid) % 16 * 8 / 64 * 8192 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(stage_tok0 + (256 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((256 + tid) % 16 * 8))), "r"((stage_len > (256 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)stage_off + (unsigned int)((384 + tid) % 16 * 8 / 64 * 8192 + ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 ^ ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(stage_tok0 + (384 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((384 + tid) % 16 * 8))), "r"((stage_len > (384 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)stage_off + (unsigned int)((512 + tid) % 16 * 8 / 64 * 8192 + ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 ^ ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(stage_tok0 + (512 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((512 + tid) % 16 * 8))), "r"((stage_len > (512 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)stage_off + (unsigned int)((640 + tid) % 16 * 8 / 64 * 8192 + ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 ^ ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(stage_tok0 + (640 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((640 + tid) % 16 * 8))), "r"((stage_len > (640 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)stage_off + (unsigned int)((768 + tid) % 16 * 8 / 64 * 8192 + ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 ^ ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(stage_tok0 + (768 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((768 + tid) % 16 * 8))), "r"((stage_len > (768 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)stage_off + (unsigned int)((896 + tid) % 16 * 8 / 64 * 8192 + ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 ^ ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(stage_tok0 + (896 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((896 + tid) % 16 * 8))), "r"((stage_len > (896 + tid) / 16) ? 16 : 0));
    if (tid_2 < 128) {
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_u_addr + (unsigned int)stage_off + (unsigned int)(tid % 2 * 8 / 64 * 8192 + (tid / 2 * 128 + tid % 2 * 8 % 64 * 2 ^ (tid / 2 * 128 + tid % 2 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(u + (((long long)(stage_tok0 + tid / 2) * (long long)num_v_heads + (long long)hv) * 128 + (long long)v0 + (long long)(tid % 2 * 8))), "r"((stage_len > tid / 2) ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;");
    #pragma unroll 1
    for (int it_chunk = 0; it_chunk < num_chunks; it_chunk++) {
        int chunk = c_begin + it_chunk;
        int tok0 = chunk_start[chunk];
        int n_valid = chunk_len[chunk];
        int next_off = 40960 - stage_off;
        if (num_chunks > it_chunk + 1) {
            int stage_tok0_0 = chunk_start[chunk + 1];
            int stage_len_1 = chunk_len[chunk + 1];
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_w_addr + (unsigned int)next_off + (unsigned int)(tid % 16 * 8 / 64 * 8192 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(w + (((long long)(stage_tok0_0 + tid / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)(tid % 16 * 8))), "r"((stage_len_1 > tid / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_w_addr + (unsigned int)next_off + (unsigned int)((128 + tid) % 16 * 8 / 64 * 8192 + ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 ^ ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(w + (((long long)(stage_tok0_0 + (128 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((128 + tid) % 16 * 8))), "r"((stage_len_1 > (128 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_w_addr + (unsigned int)next_off + (unsigned int)((256 + tid) % 16 * 8 / 64 * 8192 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(w + (((long long)(stage_tok0_0 + (256 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((256 + tid) % 16 * 8))), "r"((stage_len_1 > (256 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_w_addr + (unsigned int)next_off + (unsigned int)((384 + tid) % 16 * 8 / 64 * 8192 + ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 ^ ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(w + (((long long)(stage_tok0_0 + (384 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((384 + tid) % 16 * 8))), "r"((stage_len_1 > (384 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_w_addr + (unsigned int)next_off + (unsigned int)((512 + tid) % 16 * 8 / 64 * 8192 + ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 ^ ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(w + (((long long)(stage_tok0_0 + (512 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((512 + tid) % 16 * 8))), "r"((stage_len_1 > (512 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_w_addr + (unsigned int)next_off + (unsigned int)((640 + tid) % 16 * 8 / 64 * 8192 + ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 ^ ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(w + (((long long)(stage_tok0_0 + (640 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((640 + tid) % 16 * 8))), "r"((stage_len_1 > (640 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_w_addr + (unsigned int)next_off + (unsigned int)((768 + tid) % 16 * 8 / 64 * 8192 + ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 ^ ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(w + (((long long)(stage_tok0_0 + (768 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((768 + tid) % 16 * 8))), "r"((stage_len_1 > (768 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_w_addr + (unsigned int)next_off + (unsigned int)((896 + tid) % 16 * 8 / 64 * 8192 + ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 ^ ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(w + (((long long)(stage_tok0_0 + (896 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((896 + tid) % 16 * 8))), "r"((stage_len_1 > (896 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kn_addr + (unsigned int)next_off + (unsigned int)(tid % 16 * 8 / 64 * 8192 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(stage_tok0_0 + tid / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)(tid % 16 * 8))), "r"((stage_len_1 > tid / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kn_addr + (unsigned int)next_off + (unsigned int)((128 + tid) % 16 * 8 / 64 * 8192 + ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 ^ ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(stage_tok0_0 + (128 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((128 + tid) % 16 * 8))), "r"((stage_len_1 > (128 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kn_addr + (unsigned int)next_off + (unsigned int)((256 + tid) % 16 * 8 / 64 * 8192 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(stage_tok0_0 + (256 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((256 + tid) % 16 * 8))), "r"((stage_len_1 > (256 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kn_addr + (unsigned int)next_off + (unsigned int)((384 + tid) % 16 * 8 / 64 * 8192 + ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 ^ ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(stage_tok0_0 + (384 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((384 + tid) % 16 * 8))), "r"((stage_len_1 > (384 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kn_addr + (unsigned int)next_off + (unsigned int)((512 + tid) % 16 * 8 / 64 * 8192 + ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 ^ ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(stage_tok0_0 + (512 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((512 + tid) % 16 * 8))), "r"((stage_len_1 > (512 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kn_addr + (unsigned int)next_off + (unsigned int)((640 + tid) % 16 * 8 / 64 * 8192 + ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 ^ ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(stage_tok0_0 + (640 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((640 + tid) % 16 * 8))), "r"((stage_len_1 > (640 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kn_addr + (unsigned int)next_off + (unsigned int)((768 + tid) % 16 * 8 / 64 * 8192 + ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 ^ ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(stage_tok0_0 + (768 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((768 + tid) % 16 * 8))), "r"((stage_len_1 > (768 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kn_addr + (unsigned int)next_off + (unsigned int)((896 + tid) % 16 * 8 / 64 * 8192 + ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 ^ ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(stage_tok0_0 + (896 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((896 + tid) % 16 * 8))), "r"((stage_len_1 > (896 + tid) / 16) ? 16 : 0));
            if (tid_2 < 128) {
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(s_u_addr + (unsigned int)next_off + (unsigned int)(tid % 2 * 8 / 64 * 8192 + (tid / 2 * 128 + tid % 2 * 8 % 64 * 2 ^ (tid / 2 * 128 + tid % 2 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(u + (((long long)(stage_tok0_0 + tid / 2) * (long long)num_v_heads + (long long)hv) * 128 + (long long)v0 + (long long)(tid % 2 * 8))), "r"((stage_len_1 > tid / 2) ? 16 : 0));
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        if (tid_2 < 64) {
            float gate_val = 0.0f;
            if (tid_2 < n_valid) {
                gate_val = g_cs[(long long)(tok0 + tid_2) * (long long)num_v_heads + (long long)hv];
            }
            s_g[tid_2] = gate_val;
        }
        long long h_chunk_base = ((long long)chunk * (long long)num_v_heads + (long long)hv) * 16384;
        #pragma unroll
        for (int mb2 = 0; mb2 < 2; mb2++) {
            #pragma unroll
            for (int nb2 = 0; nb2 < 2; nb2++) {
                #pragma unroll
                for (int half_1 = 0; half_1 < 2; half_1++) {
                    int krow_1 = k_row0 + mb2 * 16 + half_1 * 8;
                    const int reg = mb2 * 2 * 4 + nb2 * 4 + half_1 * 2;
                    int vcol = nb2 * 8 + col_q;
                    {
                        __nv_bfloat16 _bval_1 = __float2bfloat16_rn(acc_h[reg]);
                        uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                        uint32_t _addr_1 = static_cast<uint32_t>((s_h_addr + (unsigned int)(krow_1 * 128 + vcol * 2 ^ (krow_1 * 128 + vcol * 2 >> 7 & 7) << 4)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_2 = __float2bfloat16_rn(acc_h[reg + 1]);
                        uint16_t _bits_2 = *(uint16_t*)&_bval_2;
                        uint32_t _addr_2 = static_cast<uint32_t>((s_h_addr + (unsigned int)(krow_1 * 128 + (vcol * 2 + 2) ^ (krow_1 * 128 + (vcol * 2 + 2) >> 7 & 7) << 4)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_2), "h"(_bits_2) : "memory");
                    }
                    __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(acc_h[reg]);
                    h_out[h_chunk_base + (long long)(v0 + vcol) * 128 + (long long)krow_1] = _cvt_bf16_0;
                    __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(acc_h[reg + 1]);
                    h_out[h_chunk_base + (long long)(v0 + vcol + 1) * 128 + (long long)krow_1] = _cvt_bf16_1;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kb = 0; kb < 8; kb++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
                : "r"(s_w_addr + (unsigned int)stage_off + (unsigned int)((kb * 16 + lane / 16 * 8) / 64 * 8192 + ((warp_1 * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 ^ ((warp_1 * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            #pragma unroll
            for (int nb3 = 0; nb3 < 1; nb3++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                    : "r"(s_h_addr + (unsigned int)((nb3 * 16 + lane / 16 * 8) / 64 * 16384 + ((kb * 16 + lane % 16) * 128 + (nb3 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb * 16 + lane % 16) * 128 + (nb3 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc_v + nb3 * 8)[0]), "=f"((acc_v + nb3 * 8)[1]), "=f"((acc_v + nb3 * 8)[2]), "=f"((acc_v + nb3 * 8)[3])
                    : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(((kb == 0) ? 0.0f : (acc_v + nb3 * 8)[0])), "f"(((kb == 0) ? 0.0f : (acc_v + nb3 * 8)[1])), "f"(((kb == 0) ? 0.0f : (acc_v + nb3 * 8)[2])), "f"(((kb == 0) ? 0.0f : (acc_v + nb3 * 8)[3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc_v + nb3 * 8 + 4)[0]), "=f"((acc_v + nb3 * 8 + 4)[1]), "=f"((acc_v + nb3 * 8 + 4)[2]), "=f"((acc_v + nb3 * 8 + 4)[3])
                    : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(((kb == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[0])), "f"(((kb == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[1])), "f"(((kb == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[2])), "f"(((kb == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[3])));
            }
        }
        float g_last = s_g[n_valid - 1];
        float decay_a = 0.0f;
        float decay_b = 0.0f;
        if (tok_row_a < n_valid) {
            float _exp2_0 = approx_exp2(g_last - s_g[tok_row_a]);
            decay_a = _exp2_0;
        }
        if (tok_row_b < n_valid) {
            float _exp2_1 = approx_exp2(g_last - s_g[tok_row_b]);
            decay_b = _exp2_1;
        }
        float vn_pair[2];
        long long vnew_a = ((long long)(tok0 + tok_row_a) * (long long)num_v_heads + (long long)hv) * 128 + (long long)v0 + (long long)col_q;
        long long vnew_b = ((long long)(tok0 + tok_row_b) * (long long)num_v_heads + (long long)hv) * 128 + (long long)v0 + (long long)col_q;
        #pragma unroll
        for (int nb4 = 0; nb4 < 2; nb4++) {
            int vcol2 = nb4 * 8 + col_q;
            vn_pair[0] = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_u) + (stage_off + (vcol2 / 64 * 8192 + (tok_row_a * 128 + vcol2 % 64 * 2 ^ (tok_row_a * 128 + vcol2 % 64 * 2 >> 7 & 7) << 4))))[0] - acc_v[nb4 * 4];
            vn_pair[1] = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_u) + (stage_off + ((vcol2 + 1) / 64 * 8192 + (tok_row_a * 128 + (vcol2 + 1) % 64 * 2 ^ (tok_row_a * 128 + (vcol2 + 1) % 64 * 2 >> 7 & 7) << 4))))[0] - acc_v[nb4 * 4 + 1];
            if (tok_row_a < n_valid) {
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(vn_pair[0 + 0], vn_pair[0 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(v_new))[vnew_a + (long long)(nb4 * 8)]) = _pk;
                }
            }
            {
                __nv_bfloat16 _bval_3 = __float2bfloat16_rn(vn_pair[0] * decay_a);
                uint16_t _bits_3 = *(uint16_t*)&_bval_3;
                uint32_t _addr_3 = static_cast<uint32_t>((s_vdec_addr + (unsigned int)(tok_row_a * 128 + vcol2 * 2 ^ (tok_row_a * 128 + vcol2 * 2 >> 7 & 7) << 4)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_3), "h"(_bits_3) : "memory");
            }
            {
                __nv_bfloat16 _bval_4 = __float2bfloat16_rn(vn_pair[1] * decay_a);
                uint16_t _bits_4 = *(uint16_t*)&_bval_4;
                uint32_t _addr_4 = static_cast<uint32_t>((s_vdec_addr + (unsigned int)(tok_row_a * 128 + (vcol2 * 2 + 2) ^ (tok_row_a * 128 + (vcol2 * 2 + 2) >> 7 & 7) << 4)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_4), "h"(_bits_4) : "memory");
            }
            vn_pair[0] = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_u) + (stage_off + (vcol2 / 64 * 8192 + (tok_row_b * 128 + vcol2 % 64 * 2 ^ (tok_row_b * 128 + vcol2 % 64 * 2 >> 7 & 7) << 4))))[0] - acc_v[nb4 * 4 + 2];
            vn_pair[1] = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_u) + (stage_off + ((vcol2 + 1) / 64 * 8192 + (tok_row_b * 128 + (vcol2 + 1) % 64 * 2 ^ (tok_row_b * 128 + (vcol2 + 1) % 64 * 2 >> 7 & 7) << 4))))[0] - acc_v[nb4 * 4 + 3];
            if (tok_row_b < n_valid) {
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(vn_pair[0 + 0], vn_pair[0 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(v_new))[vnew_b + (long long)(nb4 * 8)]) = _pk;
                }
            }
            {
                __nv_bfloat16 _bval_5 = __float2bfloat16_rn(vn_pair[0] * decay_b);
                uint16_t _bits_5 = *(uint16_t*)&_bval_5;
                uint32_t _addr_5 = static_cast<uint32_t>((s_vdec_addr + (unsigned int)(tok_row_b * 128 + vcol2 * 2 ^ (tok_row_b * 128 + vcol2 * 2 >> 7 & 7) << 4)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_5), "h"(_bits_5) : "memory");
            }
            {
                __nv_bfloat16 _bval_6 = __float2bfloat16_rn(vn_pair[1] * decay_b);
                uint16_t _bits_6 = *(uint16_t*)&_bval_6;
                uint32_t _addr_6 = static_cast<uint32_t>((s_vdec_addr + (unsigned int)(tok_row_b * 128 + (vcol2 * 2 + 2) ^ (tok_row_b * 128 + (vcol2 * 2 + 2) >> 7 & 7) << 4)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_6), "h"(_bits_6) : "memory");
            }
        }
        __syncthreads();
        float _exp2_2 = approx_exp2(g_last);
        const float2 _scale2_7 = {_exp2_2, _exp2_2};
        #pragma unroll
        for (int _ls = 0; _ls < 8; _ls++)
            mul_f32x2_inplace(&reinterpret_cast<float2*>(acc_h)[_ls], _scale2_7);
        #pragma unroll
        for (int kb2 = 0; kb2 < 4; kb2++) {
            #pragma unroll
            for (int mb3 = 0; mb3 < 2; mb3++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag_mn[0]), "=r"(a_frag_mn[1]), "=r"(a_frag_mn[2]), "=r"(a_frag_mn[3])
                    : "r"(s_kn_addr + (unsigned int)stage_off + (unsigned int)((warp_1 * 32 + mb3 * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((kb2 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (warp_1 * 32 + mb3 * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((kb2 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (warp_1 * 32 + mb3 * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                #pragma unroll
                for (int nb5 = 0; nb5 < 1; nb5++) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                        : "r"(s_vdec_addr + (unsigned int)((nb5 * 16 + lane / 16 * 8) / 64 * 8192 + ((kb2 * 16 + lane % 16) * 128 + (nb5 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb2 * 16 + lane % 16) * 128 + (nb5 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc_h + mb3 * 2 * 4 + nb5 * 8)[0]), "+f"((acc_h + mb3 * 2 * 4 + nb5 * 8)[1]), "+f"((acc_h + mb3 * 2 * 4 + nb5 * 8)[2]), "+f"((acc_h + mb3 * 2 * 4 + nb5 * 8)[3])
                        : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc_h + mb3 * 2 * 4 + nb5 * 8 + 4)[0]), "+f"((acc_h + mb3 * 2 * 4 + nb5 * 8 + 4)[1]), "+f"((acc_h + mb3 * 2 * 4 + nb5 * 8 + 4)[2]), "+f"((acc_h + mb3 * 2 * 4 + nb5 * 8 + 4)[3])
                        : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                }
            }
        }
        __syncthreads();
        stage_off = next_off;
    }
    if (store_final_state != 0) {
        #pragma unroll
        for (int mb4 = 0; mb4 < 2; mb4++) {
            #pragma unroll
            for (int nb6 = 0; nb6 < 2; nb6++) {
                #pragma unroll
                for (int half2 = 0; half2 < 2; half2++) {
                    int krow2 = k_row0 + mb4 * 16 + half2 * 8;
                    {
                        float2 _v2 = make_float2(acc_h[mb4 * 2 * 4 + nb6 * 4 + half2 * 2 + 0], acc_h[mb4 * 2 * 4 + nb6 * 4 + half2 * 2 + 1]);
                        *reinterpret_cast<float2*>(final_state + state_base + (long long)krow2 * 128 + (long long)v0 + (long long)(nb6 * 8) + (long long)col_q) = _v2;
                    }
                }
            }
        }
    }
}

} // extern "C"
