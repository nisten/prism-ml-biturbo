#pragma once

// Fast Walsh-Hadamard Transform (FWHT) infrastructure for TBQ4_0 CUDA kernels.
//
// Replaces the Householder Q^T matvec (O(128²)) with FWHT (O(128×7)):
//   Forward:  y = D_s2 · H_norm · D_s1 · x
//   Inverse:  x = D_s1 · H_norm · D_s2 · y
// where H_norm = H/√128 (orthogonal), D_s1/D_s2 are diagonal sign matrices.
//
// Both CPY (dequant TBQ4_0→F32) and SET_ROWS (quant F32→TBQ4_0) must use
// the same transform (inverse/forward respectively) for correctness.

#include "ggml-common.h"

// ─── Device constants ────────────────────────────────────────────────────────

// Sign arrays: ±1.0f, 128 elements each.
// Generated from splitmix64 seed 0x517cc1b727220a95ULL (same as Householder seed).
__device__ __constant__ float d_tbq_wht_s1[128];
__device__ __constant__ float d_tbq_wht_s2[128];

// Codebook values divided by √256 (= /16), ready for dequant output.
__device__ __constant__ float d_tbq4_codebook_scaled[16];

// Lloyd-Max 4-bit quantization boundaries (15 values for 16 bins).
__device__ __constant__ float d_tbq4_boundaries[15];

// ─── Host initialization ─────────────────────────────────────────────────────

static bool g_tbq_wht_initialized = false;

static inline uint64_t tbq_splitmix64(uint64_t * state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void tbq_wht_init(void) {
    if (g_tbq_wht_initialized) return;

    // Generate sign arrays from fixed seed (must never change — determines rotation)
    float s1[128], s2[128];
    uint64_t state = 0x517cc1b727220a95ULL;
    for (int i = 0; i < 128; i++) {
        s1[i] = (tbq_splitmix64(&state) & 1) ? 1.0f : -1.0f;
    }
    for (int i = 0; i < 128; i++) {
        s2[i] = (tbq_splitmix64(&state) & 1) ? 1.0f : -1.0f;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_tbq_wht_s1, s1, sizeof(s1)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_tbq_wht_s2, s2, sizeof(s2)));

    // Codebook / sqrt(256): dequant output = codebook_scaled[idx] * norm
    const float scale_down = 1.0f / 16.0f;  // 1/sqrt(256)
    float cb[16] = {
        -2.7326f * scale_down, -2.0690f * scale_down, -1.6180f * scale_down, -1.2562f * scale_down,
        -0.9424f * scale_down, -0.6568f * scale_down, -0.3881f * scale_down, -0.1284f * scale_down,
         0.1284f * scale_down,  0.3881f * scale_down,  0.6568f * scale_down,  0.9424f * scale_down,
         1.2562f * scale_down,  1.6180f * scale_down,  2.0690f * scale_down,  2.7326f * scale_down,
    };
    CUDA_CHECK(cudaMemcpyToSymbol(d_tbq4_codebook_scaled, cb, sizeof(cb)));

    // Lloyd-Max 4-bit quantization boundaries
    float bnd[15] = {
        -2.4008f, -1.8435f, -1.4371f, -1.0993f,
        -0.7996f, -0.5225f, -0.2583f,  0.0000f,
         0.2583f,  0.5225f,  0.7996f,  1.0993f,
         1.4371f,  1.8435f,  2.4008f,
    };
    CUDA_CHECK(cudaMemcpyToSymbol(d_tbq4_boundaries, bnd, sizeof(bnd)));

    g_tbq_wht_initialized = true;
}

// ─── Device quantize helper ───────────────────────────────────────────────────

// Map a float value to 4-bit index (0..15) using Lloyd-Max boundaries.
static __device__ __forceinline__ uint8_t tbq_quantize_4bit(float val) {
    uint8_t idx = 0;
    #pragma unroll
    for (int k = 0; k < 15; k++) {
        if (val > d_tbq4_boundaries[k]) idx = (uint8_t)(k + 1);
    }
    return idx;
}
