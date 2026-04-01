# prism-ml-biturbo

**TurboQuant 4-bit KV cache with full CUDA support for PrismML's 1-bit llama.cpp fork**

Fork of [PrismML/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) adding:
1. dp4a integer matmul kernel for Q1_0_g128 on non-RTX Turing GPUs
2. TBQ4_0 (TurboQuant 4-bit) KV cache quantization with CUDA quantize + dequantize
3. Fast Walsh-Hadamard Transform (FWHT) for O(n log n) rotation instead of O(n^2) Householder

Tested on GTX 1660 Ti (Turing, sm_75, 6 GB GDDR6, NO tensor cores) running
PrismML's Bonsai-8B (Qwen3-8B, Q1_0_g128, 1.08 GB model weights).

```
+---------------------------------------------------------------------+
|                        BENCHMARK RESULTS                            |
|                   GTX 1660 Ti / Bonsai-8B 1-bit                     |
+---------------------------+------------+---------+------------------+
| Configuration             | Prompt t/s | Gen t/s | KV Compression   |
+---------------------------+------------+---------+------------------+
| f16 KV (baseline)         |       86.1 |    53.3 | 1.0x  (512 B)   |
| TBQ4_0 CPU fallback (bug) |        --- |     0.9 | 3.94x (130 B)   |
| TBQ4_0 CPU KV  (-nkvo)   |       24.0 |     9.7 | 3.94x (130 B)   |
| TBQ4_0 full GPU KV        |        --- |    34.4 | 3.94x (130 B)   |
+---------------------------+------------+---------+------------------+
  "KV Compression" = bytes per 256 elements (one TBQ4_0 block)
  "full GPU KV"    = FWHT SET_ROWS, no -nkvo flag needed
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [What This Fork Adds](#what-this-fork-adds)
- [The Model: Bonsai-8B / Q1_0_g128](#the-model-bonsai-8b--q1_0_g128)
- [Phase 1: dp4a Kernel for Non-RTX Turing](#phase-1-dp4a-kernel-for-non-rtx-turing)
- [Phase 2: TurboQuant TBQ4_0 KV Cache](#phase-2-turboquant-tbq4_0-kv-cache)
- [Phase 3: FWHT + Full GPU KV Cache](#phase-3-fwht--full-gpu-kv-cache)
- [Bugs Found and Fixed](#bugs-found-and-fixed)
- [File Changes Summary](#file-changes-summary)
- [Known Issues and TODOs](#known-issues-and-todos)
- [How TBQ4_0 Works](#how-tbq4_0-works)
- [FWHT Math](#fwht-math)
- [Credits and References](#credits-and-references)
- [Original llama.cpp README](#original-llamacpp-readme)

---

## Quick Start

```bash
# Clone
git clone https://github.com/nisten/prism-ml-biturbo.git
cd prism-ml-biturbo

# Build (IMPORTANT: always pin your CUDA architecture)
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75    # <-- change 75 to your GPU's sm_XX
cmake --build build --config Release -j$(nproc)

# Run with TBQ4_0 KV cache (3.94x compression)
./build/bin/llama-cli \
  -m /path/to/Bonsai-8B.gguf \
  -c 12000 -ngl 99 -fa on -t 4 --mlock \
  --chat-template chatml -cnv \
  -p "You are a helpful assistant." --temp 0.5 \
  -ctk tbq4_0 -ctv tbq4_0

# Run with default f16 KV cache (no compression, faster)
./build/bin/llama-cli \
  -m /path/to/Bonsai-8B.gguf \
  -c 12000 -ngl 99 -fa on -t 4 --mlock \
  --chat-template chatml -cnv \
  -p "You are a helpful assistant."
```

**Build requirement**: Always specify `-DCMAKE_CUDA_ARCHITECTURES=XX` explicitly.
CUDA 13.x default architecture auto-detection can produce broken PTX that silently
generates garbage output (we lost hours to this -- see [Bugs Found](#bugs-found-and-fixed)).

---

## What This Fork Adds

24 files changed, +1235 lines, -26 lines over PrismML base.

```
CHANGES vs PrismML upstream (prism branch)
==========================================

New files:
  ggml/src/ggml-cuda/tbq-wht.cuh      FWHT infrastructure (sign arrays, codebook, init)
  ggml/src/ggml-turboq-tables.h        Lloyd-Max codebooks for 2/3/4-bit quantization
  ggml/src/ggml-turboq.h               TurboQuant rotation API header
  ggml/src/ggml-turboq.c               Full rotation + quantize/dequantize (688 lines)

Modified CUDA files:
  ggml/src/ggml-cuda/cpy.cu            TBQ4_0->F32 CUDA dequant kernel (FWHT inverse)
  ggml/src/ggml-cuda/cpy-utils.cuh     F32->TBQ4_0 CUDA quant function (FWHT forward)
  ggml/src/ggml-cuda/set-rows.cu       TBQ4_0 dispatch in SET_ROWS operation
  ggml/src/ggml-cuda/ggml-cuda.cu      TBQ4_0 registered in CPY + SET_ROWS supports_op
  ggml/src/ggml-cuda/mmq.cuh           dp4a kernel for Q1_0_g128 (non-RTX Turing)

Modified core files:
  ggml/include/ggml.h                  GGML_TYPE_TBQ3_0 (42), TBQ4_0 (43), COUNT=44
  ggml/src/ggml-common.h               block_tbq3_0 (98B), block_tbq4_0 (130B) structs
  ggml/src/ggml.c                      type_traits, ftype mapping, quantize dispatch
  ggml/src/ggml-quants.h               quantize/dequantize declarations
  ggml/src/CMakeLists.txt              turboq source files added to build

Modified CPU files:
  ggml/src/ggml-cpu/ggml-cpu.c         CPU type_traits for TBQ (vec_dot + from_float)
  ggml/src/ggml-cpu/ops.cpp            TBQ CPU operations support
  ggml/src/ggml-cpu/quants.c           vec_dot_tbq3_0_q8_K, vec_dot_tbq4_0_q8_K
  ggml/src/ggml-cpu/quants.h           vec_dot declarations

Modified llama files:
  include/llama.h                      LLAMA_FTYPE_MOSTLY_TBQ3_0 / TBQ4_0
  src/llama-model-loader.cpp           ftype string + type mapping
  src/llama-kv-cache.cpp               3D view for TBQ K/V (block_size > head_dim)
  src/llama-graph.cpp                  TBQ cast+reshape before attention permute
  src/llama-context.cpp                TBQ KV cache initialization
  common/arg.cpp                       TBQ3_0/TBQ4_0 in kv_cache_types
```

---

## The Model: Bonsai-8B / Q1_0_g128

PrismML's Bonsai-8B is Qwen3-8B quantized to 1-bit (Q1_0_g128):
- 8.19 billion parameters compressed to 1.08 GB (1.126 bits per weight)
- Architecture: 36 layers, GQA (32 query heads / 8 KV heads), SwiGLU FFN
- Each weight is a single bit (sign only), groups of 128 share one FP16 scale
- Block format: 2 bytes (FP16 scale) + 16 bytes (128 sign bits) = 18 bytes per 128 weights
- bit=1 means +scale, bit=0 means -scale

```
Q1_0_g128 block layout (18 bytes per 128 weights):
+-------+---------------------------------------------------+
| scale |                  128 sign bits                     |
| FP16  |              (16 bytes = 128 bits)                 |
| 2B    |  bit[i]=1 -> +scale,  bit[i]=0 -> -scale          |
+-------+---------------------------------------------------+
```

Standard llama.cpp does not support Q1_0_g128. This format requires PrismML's fork
which adds dedicated CUDA and CPU kernels for 1-bit inference.

---

## Phase 1: dp4a Kernel for Non-RTX Turing

**Problem**: PrismML's Q1_0_g128 CUDA kernels only support the MMA (tensor core) path.
The GTX 1660 Ti is Turing (sm_75) but has NO tensor cores -- MMA instructions are
emulated in software, producing a "suboptimal performance" warning and leaving
performance on the table.

**Root cause**: `TURING_MMA_AVAILABLE` is defined for all `__CUDA_ARCH__ >= 750`,
which incorrectly includes GTX 1660 Ti. Q1_0_g128 has dp4a intentionally disabled
in `mmq.cuh` line 513 (`NO_DEVICE_CODE` for the dp4a path).

**Key insight**: After bit-unpacking, Q1_0_g128 data is identical to Q8_0 format
(signed int8 values). We can replicate the scale 4x during tile load and reuse
`vec_dot_q8_0_q8_1_dp4a` unchanged.

**Changes** (`ggml/src/ggml-cuda/mmq.cuh`):
1. Added `GGML_TYPE_Q1_0_g128` to `mmq_get_dp4a_tile_x_sizes()` returning `MMQ_DP4A_TXS_Q8_0`
2. Rewrote `load_tiles_q1_0_g128` with dual MMA/dp4a branches
3. Updated `mmq_type_traits<Q1_0_g128>::vec_dot_dp4a` from `disabled` to `vec_dot_q8_0_q8_1_dp4a`

```
Results (GTX 1660 Ti, sm_75):
+------------+-------------+--------------+---------+
| Metric     | MMA (stock) | dp4a (ours)  | Change  |
+------------+-------------+--------------+---------+
| Generation |  49.2 t/s   |  54.1 t/s    | +10%    |
| Prompt     |  85.5 t/s   |  85.0 t/s    | ~same   |
+------------+-------------+--------------+---------+
  Prompt is memory-bandwidth bound, generation is compute bound.
  dp4a removes the tensor core emulation overhead.
```

---

## Phase 2: TurboQuant TBQ4_0 KV Cache

**Goal**: Compress the KV cache using TurboQuant 4-bit quantization (TBQ4_0) to fit
longer contexts in limited VRAM. The 1-bit model weights are only 1.08 GB, but at
16K context the f16 KV cache alone consumes ~575 MB. With 3.94x compression, TBQ4_0
reduces this to ~146 MB.

**What is TBQ4_0?** Walsh-Hadamard Transform rotation + Lloyd-Max scalar 4-bit
quantization, achieving near-zero quality loss at 3.94x compression. Each 256-element
block is stored as 130 bytes (128 nibble-packed quantized values + 2 bytes FP16 norm).

**Implementation** (ported from elusznik's PR #21089 CPU implementation):
- Registered `GGML_TYPE_TBQ4_0` (type ID 43) in ggml.h
- Added `block_tbq4_0` struct (130 bytes: 128 qs + 2 d) in ggml-common.h
- Ported full rotation + quantize/dequantize (688 lines) in ggml-turboq.c
- Added Lloyd-Max codebooks in ggml-turboq-tables.h
- Wired into KV cache system (llama-kv-cache, llama-graph, llama-context)
- Added `-ctk tbq4_0 -ctv tbq4_0` CLI support via common/arg.cpp
- Added CPU vec_dot for flash attention fallback

**Initial results**: TBQ4_0 produced coherent output with the CPU quantize/dequantize
path, but was bottlenecked by the GGML scheduler silently falling back to CPU for the
dequantize operation (see [Bugs Found](#bugs-found-and-fixed) for the full story).

---

## Phase 3: FWHT + Full GPU KV Cache

**Motivation**: spiritbuun's llama-cpp-turboquant-cuda demonstrated that using
Fast Walsh-Hadamard Transform (FWHT) instead of Householder QR rotation achieves
99.6% prefill / 97.5% decode speed vs f16 baseline on RTX 3090. FWHT is O(n log n)
vs Householder O(n^2), uses 256 bytes of sign arrays instead of a 64 KB rotation matrix,
and has a butterfly structure that maps naturally to GPU parallelism.

**The two CUDA operations needed**:

```
                    +-----------+
   F32 values --->  | SET_ROWS  | ---> TBQ4_0 blocks in GPU KV cache
   (new tokens)     | (quant)   |      (FWHT forward rotation + Lloyd-Max 4-bit)
                    +-----------+

                    +-----------+
   TBQ4_0 blocks -> |    CPY    | ---> F32 values for attention computation
   (from KV cache)  | (dequant) |      (FWHT inverse rotation + codebook lookup)
                    +-----------+
```

**New file: `ggml/src/ggml-cuda/tbq-wht.cuh`**

Shared FWHT infrastructure used by both operations:
- Device constant sign arrays: `d_tbq_wht_s1[128]`, `d_tbq_wht_s2[128]` (each +/-1.0f)
- Device constant scaled codebook: `d_tbq4_codebook_scaled[16]` (codebook / sqrt(256))
- Device constant boundaries: `d_tbq4_boundaries[15]` (Lloyd-Max decision boundaries)
- `tbq_wht_init()`: lazy host init, generates sign arrays from splitmix64 PRNG with
  fixed seed `0x517cc1b727220a95ULL`, uploads via `cudaMemcpyToSymbol`
- `tbq_quantize_4bit()`: device function mapping float to 4-bit index via boundary search

**CPY kernel (dequantize TBQ4_0 -> F32)** in `ggml/src/ggml-cuda/cpy.cu`:
- 256 threads per block (one block per TBQ4_0 block)
- Threads 0-127 handle sub-block 0, threads 128-255 handle sub-block 1
- Step 1: Each thread unpacks its nibble from qs[], looks up scaled codebook value
- Step 2: FWHT inverse via shared memory (7 butterfly passes with __syncthreads)
- Step 3: Multiply by FP16 norm, write to F32 output

**SET_ROWS quantize function** in `ggml/src/ggml-cuda/cpy-utils.cuh`:
- `quantize_f32_tbq4_0_block()`: single-threaded F32[256] -> block_tbq4_0
- Called from the existing `k_set_rows_quant` template (one CUDA thread per output block)
- Computes L2 norm, normalizes to unit vector
- Applies FWHT forward rotation (s1 multiply, 7 butterfly passes, normalize, s2 multiply)
- Quantizes rotated values via boundary comparison, nibble-packs into qs[]
- Stores FP16 norm in the block's d field

---

## Bugs Found and Fixed

### Bug 1: CUDA 13.x Default Architecture Produces Garbage

**Symptom**: ALL inference produced garbage output at 0.1-0.3 t/s, regardless of model,
quantization type, or branch. Even the unmodified PrismML base branch was affected.

**Root cause**: CUDA 13.1 default architecture auto-detection compiles PTX for all
supported architectures. The JIT compiler then selects PTX for the GTX 1660 Ti but
picks a suboptimal or incorrect target. This affects the entire computation, not just
specific quantization types.

**Resolution**: ALWAYS specify `-DCMAKE_CUDA_ARCHITECTURES=75` (or your GPU's sm_XX)
when building. This was discovered through a Kepner-Tregoe IS/IS-NOT analysis:

```
 IS                       IS NOT                 DISTINCTION
 -----------------------  ---------------------- -------------------------
 ALL inference garbage    Code bug               Even pristine PrismML base
 (0.1 t/s, broken)                              broken with default cmake

 Default cmake flags      cmake with             -DCMAKE_CUDA_ARCH=75
 (-DGGML_CUDA=ON only)    -DCMAKE_CUDA_ARCH=75   works (53.3 t/s, coherent)

 After rm -rf build +     Earlier builds         Earlier build had sm_75
 rebuild without flag      that worked fine        pinned from prior cmake
```

### Bug 2: Silent CPU Fallback for TBQ4_0 Dequantize

**Symptom**: TBQ4_0 KV cache produced 0.9 t/s generation (60x slower than expected).
No error messages, no crashes. The model appeared to work but was unusably slow.

**Root cause**: The GGML scheduler's Pass 3 checks `ggml_backend_cuda_device_supports_op()`
for each operation. `GGML_OP_CPY` with `TBQ4_0->F32` was NOT listed in the CUDA backend's
supported operations. The scheduler silently assigned dequantization to CPU.

Every decode token: CPU dequantized KV data -> transferred F32 over PCIe -> GPU ran attention.
The PCIe round-trip dominated latency.

**Fix**: Registered `TBQ4_0->F32` in `ggml_backend_cuda_device_supports_op()` for
`GGML_OP_CPY` and implemented `cpy_tbq4_0_f32_kernel` on CUDA.

**Result**: 0.9 t/s -> 9.7 t/s (10.8x speedup). Still limited by -nkvo (KV on CPU).

### Bug 3: SIGABRT Without -nkvo (SET_ROWS Not Implemented)

**Symptom**: Without `-nkvo` flag, llama-cli crashes with:
`cache_k_l0 (view) in a buffer (CUDA0) that cannot run the operation (SET_ROWS)`

**Root cause**: When KV cache is on GPU (no -nkvo), new token K/V values must be
quantized on the GPU via `GGML_OP_SET_ROWS`. TBQ4_0 was not in the SET_ROWS
dispatch table or the supports_op check.

**Fix**: Implemented `quantize_f32_tbq4_0_block()` using FWHT forward rotation,
added dispatch in `set-rows.cu`, registered in `ggml-cuda.cu` supports_op.

**Result**: 9.7 t/s -> 34.4 t/s (3.5x speedup). Full GPU KV cache, no -nkvo needed.

### Bug 4: smem Round-Trip in Dequant Kernel (Security Review)

**Symptom**: No runtime symptoms observed, but a security review flagged that
the final step in `cpy_tbq4_0_f32_kernel` wrote to shared memory and then read
it back without a `__syncthreads()` between them.

**Analysis**: Each thread only reads from its own shared memory index, so there
is no actual cross-thread dependency. The CUDA memory model guarantees intra-thread
ordering. However, computing the final multiply in a register and writing directly
to global memory is cleaner and avoids any theoretical concern.

**Fix**: Changed from `sub[lid] *= s1[lid]; dst = sub[lid] * norm;` to
`float out = sub[lid] * s1[lid]; dst = out * norm;` (register-only, no smem write-back).

---

## File Changes Summary

```
ggml/src/ggml-cuda/tbq-wht.cuh         86 lines  NEW   FWHT constants + init + quantize helper
ggml/src/ggml-turboq.c                 688 lines  NEW   CPU rotation + quantize/dequantize
ggml/src/ggml-turboq.h                  25 lines  NEW   Rotation API header
ggml/src/ggml-turboq-tables.h           35 lines  NEW   Lloyd-Max codebooks
ggml/src/ggml-cuda/cpy.cu              +93 lines  MOD   FWHT inverse dequant kernel
ggml/src/ggml-cuda/cpy-utils.cuh       +59 lines  MOD   FWHT forward quant function
ggml/src/ggml-cuda/set-rows.cu         +11 lines  MOD   TBQ4_0 dispatch
ggml/src/ggml-cuda/ggml-cuda.cu         +6 lines  MOD   TBQ4_0 in supports_op (CPY + SET_ROWS)
ggml/src/ggml-cuda/mmq.cuh             +39 lines  MOD   dp4a for Q1_0_g128
ggml/src/ggml-common.h                 +17 lines  MOD   block_tbq3_0, block_tbq4_0 structs
ggml/include/ggml.h                     +6 lines  MOD   Type enums TBQ3_0=42, TBQ4_0=43
ggml/src/ggml.c                        +20 lines  MOD   type_traits + ftype mapping
ggml/src/ggml-quants.h                  +6 lines  MOD   Quantize/dequantize declarations
ggml/src/ggml-cpu/quants.c             +46 lines  MOD   CPU vec_dot for TBQ
ggml/src/ggml-cpu/ggml-cpu.c           +13 lines  MOD   CPU type_traits for TBQ
ggml/src/ggml-cpu/ops.cpp              +14 lines  MOD   CPU TBQ operations
ggml/src/CMakeLists.txt                 +3 lines  MOD   Build system
include/llama.h                         +2 lines  MOD   LLAMA_FTYPE entries
src/llama-model-loader.cpp              +4 lines  MOD   ftype string mapping
src/llama-kv-cache.cpp                 +18 lines  MOD   3D view for TBQ K/V
src/llama-graph.cpp                    +36 lines  MOD   TBQ cast+reshape
src/llama-context.cpp                  +30 lines  MOD   TBQ KV init
common/arg.cpp                          +2 lines  MOD   CLI --cache-type support
```

---

## Known Issues and TODOs

**Known issues**:
- CPU-only inference with TBQ4_0 produces garbage (0.3 t/s). The CPU flash attention
  path has issues with TBQ4_0 vec_dot. GPU inference works correctly.
- The runtime warning "suboptimal performance due to a lack of tensor cores" still
  appears on GTX 1660 Ti. This is a cosmetic check in ggml_cuda_init that tests for
  tensor cores at runtime, not affected by our dp4a kernel being active. Harmless.
- TBQ3_0 (3-bit TurboQuant) type is registered but NOT implemented in CUDA.
  Only TBQ4_0 has complete CUDA support.
- The FWHT rotation is different from the Householder QR rotation used in the CPU
  reference implementation (ggml-turboq.c). Both are valid random orthogonal rotations
  but they are NOT interchangeable. The GPU path (FWHT) and CPU path (Householder)
  produce different quantized blocks for the same input. Since KV cache is ephemeral
  (rebuilt each session), this is not a correctness issue -- but mixing GPU quantize
  with CPU dequantize (or vice versa) would produce garbage.

**TODOs**:
- [ ] Benchmark 16K context with full GPU KV (measure VRAM usage)
- [ ] Test maximum context length that fits in 6 GB VRAM
- [ ] Benchmark on other GPUs (RTX 3060, 4060, A100)
- [ ] Upstream the dp4a Q1_0_g128 fix to PrismML
- [ ] Consider multi-threaded CUDA quantize kernel for SET_ROWS (currently single-threaded
      per block via k_set_rows_quant template -- fine for decode but slow for large batch prefill)
- [ ] Port FWHT to the CPU path (replace Householder) for consistency
- [ ] TBQ3_0 CUDA implementation (2.5x compression, higher quality)
- [ ] Profile the FWHT butterfly passes -- the 7 __syncthreads calls per dequant block
      may be optimizable with warp-level primitives for the first 5 passes (h <= 32)

---

## How TBQ4_0 Works

TBQ4_0 quantizes 256-element vectors (one per KV cache block) to 130 bytes:

```
Quantize (F32 -> TBQ4_0):
  1. Compute L2 norm:  norm = ||x||
  2. Normalize:        u = x / norm
  3. Split into two 128-element sub-blocks
  4. For each sub-block:
     a. Forward FWHT rotation:  y = D_s2 * H_norm * D_s1 * u
     b. Scale up:               z = y * sqrt(256)
     c. Lloyd-Max 4-bit quantize: idx[i] = argmin_k |z[i] - codebook[k]|
     d. Nibble pack:            qs[j/2] = idx[j] | (idx[j+1] << 4)
  5. Store FP16 norm

Dequantize (TBQ4_0 -> F32):
  1. Unpack nibbles:   idx[i] = qs[i/2] & 0x0F or qs[i/2] >> 4
  2. Codebook lookup:  y[i] = codebook[idx[i]] / sqrt(256)
  3. For each sub-block:
     a. Inverse FWHT:  u = D_s1 * H_norm * D_s2 * y
  4. Scale by norm:    x[i] = u[i] * norm

Block layout (130 bytes):
+----------------------------------------------------------+------+
|                  qs[128] (nibble-packed)                   |  d   |
|  256 values packed as 128 bytes (low nibble + high nibble) | FP16 |
+----------------------------------------------------------+------+
  offset 0                                                    128
```

The Lloyd-Max codebook for 4-bit (16 levels) optimized for standard normal distribution:

```
Index:  0       1       2       3       4       5       6       7
Value: -2.7326 -2.0690 -1.6180 -1.2562 -0.9424 -0.6568 -0.3881 -0.1284

Index:  8       9      10      11      12      13      14      15
Value:  0.1284  0.3881  0.6568  0.9424  1.2562  1.6180  2.0690  2.7326
```

---

## FWHT Math

The Fast Walsh-Hadamard Transform provides O(n log n) random orthogonal rotation
using only element-wise sign flips and butterfly additions. No matrix storage needed.

```
Forward transform:   y = D_s2 * H_norm * D_s1 * x
Inverse transform:   x = D_s1 * H_norm * D_s2 * y

Where:
  D_s1, D_s2 = diagonal matrices of +/-1 (random signs from splitmix64 PRNG)
  H_norm     = H / sqrt(n)  (normalized Hadamard matrix, n=128)
  H          = 128x128 Walsh-Hadamard matrix (defined recursively)

Properties:
  - H * H = n * I       (Hadamard is self-inverse up to scale)
  - D_si * D_si = I     (sign matrices are self-inverse)
  - Forward(Inverse(x)) = x  (exact round-trip, proof below)

Proof of inverse:
  D_s1 * H_norm * D_s2 * (D_s2 * H_norm * D_s1 * x)
  = D_s1 * H_norm * (D_s2 * D_s2) * H_norm * D_s1 * x
  = D_s1 * H_norm * I * H_norm * D_s1 * x
  = D_s1 * (H_norm * H_norm) * D_s1 * x
  = D_s1 * (H*H / n) * D_s1 * x
  = D_s1 * (n*I / n) * D_s1 * x
  = D_s1 * D_s1 * x
  = x

The butterfly structure of WHT for n=128:
  7 passes (log2(128) = 7), each pass with stride h = 1, 2, 4, 8, 16, 32, 64
  Each butterfly:  a' = a + b,  b' = a - b

  for h in {1, 2, 4, 8, 16, 32, 64}:
    for j in {0, 1, ..., 127}:
      if (j & h) == 0:
        a = buf[j],  b = buf[j | h]
        buf[j] = a + b,  buf[j | h] = a - b

Operation count: 128 * 7 = 896 additions/subtractions
vs. Householder matvec: 128 * 128 = 16384 multiply-adds (18x more)
```

The sign arrays are generated deterministically from a fixed seed so that every
process produces the same rotation. This is critical -- if the quantize and dequantize
paths use different sign arrays, the inverse rotation is wrong and output is garbage.

```c
uint64_t state = 0x517cc1b727220a95ULL;  // fixed seed, must never change
for (int i = 0; i < 128; i++)
    s1[i] = (splitmix64_next(&state) & 1) ? +1.0f : -1.0f;
for (int i = 0; i < 128; i++)
    s2[i] = (splitmix64_next(&state) & 1) ? +1.0f : -1.0f;
```

---

## Credits and References

- [PrismML/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) -- base fork with Q1_0_g128 1-bit kernels
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) -- upstream llama.cpp
- [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) -- FWHT CUDA TurboQuant reference (inspired our FWHT implementation)
- [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) -- Metal+CUDA+HIP TurboQuant
- [elusznik's PR #21089](https://github.com/ggml-org/llama.cpp/pull/21089) -- CPU TBQ3_0/TBQ4_0 (our CPU path is based on this)
- TurboQuant paper: Walsh-Hadamard rotation + Lloyd-Max scalar quantization for KV cache compression

---

## Original llama.cpp README

*The original llama.cpp README follows below.*

---

# llama.cpp

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp)](https://github.com/ggml-org/llama.cpp/releases)
[![Server](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml/badge.svg)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)

[Manifesto](https://github.com/ggml-org/llama.cpp/discussions/205) / [ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)

LLM inference in C/C++

## Recent API changes

- [Changelog for `libllama` API](https://github.com/ggml-org/llama.cpp/issues/9289)
- [Changelog for `llama-server` REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

## Hot topics

- **[guide : using the new WebUI of llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/16938)**
- [guide : running gpt-oss with llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/15396)
- [[FEEDBACK] Better packaging for llama.cpp to support downstream consumers 🤗](https://github.com/ggml-org/llama.cpp/discussions/15313)
- Support for the `gpt-oss` model with native MXFP4 format has been added | [PR](https://github.com/ggml-org/llama.cpp/pull/15091) | [Collaboration with NVIDIA](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss) | [Comment](https://github.com/ggml-org/llama.cpp/discussions/15095)
- Multimodal support arrived in `llama-server`: [#12898](https://github.com/ggml-org/llama.cpp/pull/12898) | [documentation](./docs/multimodal.md)
- VS Code extension for FIM completions: https://github.com/ggml-org/llama.vscode
- Vim/Neovim plugin for FIM completions: https://github.com/ggml-org/llama.vim
- Hugging Face Inference Endpoints now support GGUF out of the box! https://github.com/ggml-org/llama.cpp/discussions/9669
- Hugging Face GGUF editor: [discussion](https://github.com/ggml-org/llama.cpp/discussions/9268) | [tool](https://huggingface.co/spaces/CISCai/gguf-editor)

----

## Quick start

Getting started with llama.cpp is straightforward. Here are several ways to install it on your machine:

- Install `llama.cpp` using [brew, nix or winget](docs/install.md)
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed, you'll need a model to work with. Head to the [Obtaining and quantizing models](#obtaining-and-quantizing-models) section to learn more.

Example command:

```sh
# Use a local model file
llama-cli -m my_model.gguf

# Or download and run a model directly from Hugging Face
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF

# Launch OpenAI-compatible API server
llama-server -hf ggml-org/gemma-3-1b-it-GGUF
```

## Description

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide
range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is the main playground for developing new features for the [ggml](https://github.com/ggml-org/ggml) library.

<details>
<summary>Models</summary>

Typically finetunes of the base models below are supported as well.

Instructions for adding support for new models: [HOWTO-add-model.md](docs/development/HOWTO-add-model.md)

#### Text-only

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [x] [Jamba](https://huggingface.co/ai21labs)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [X] [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [X] [BERT](https://github.com/ggml-org/llama.cpp/pull/5423)
- [X] [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan) + [derivations](https://huggingface.co/hiyouga/baichuan-7b-sft)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggml-org/llama.cpp/pull/3187)
- [X] [Refact](https://huggingface.co/smallcloudai/Refact-1_6B-fim)
- [X] [MPT](https://github.com/ggml-org/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggml-org/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)
- [X] [StableLM models](https://huggingface.co/stabilityai)
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [PLaMo-13B](https://github.com/ggml-org/llama.cpp/pull/3557)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [PhiMoE](https://github.com/ggml-org/llama.cpp/pull/11003)
- [x] [GPT-2](https://huggingface.co/gpt2)
- [x] [Orion 14B](https://github.com/ggml-org/llama.cpp/pull/5118)
- [x] [InternLM2](https://huggingface.co/models?search=internlm2)
- [x] [CodeShell](https://github.com/WisdomShell/codeshell)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/keyfan/grok-1-hf)
- [x] [Xverse](https://huggingface.co/models?search=xverse)
- [x] [Command-R models](https://huggingface.co/models?search=CohereForAI/c4ai-command-r)
- [x] [SEA-LION](https://huggingface.co/models?search=sea-lion)
- [x] [GritLM-7B](https://huggingface.co/GritLM/GritLM-7B) + [GritLM-8x7B](https://huggingface.co/GritLM/GritLM-8x7B)
- [x] [OLMo](https://allenai.org/olmo)
- [x] [OLMo 2](https://allenai.org/olmo)
- [x] [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- [x] [Granite models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330)
- [x] [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) + [Pythia](https://github.com/EleutherAI/pythia)
- [x] [Snowflake-Arctic MoE](https://huggingface.co/collections/Snowflake/arctic-66290090abe542894a5ac520)
- [x] [Smaug](https://huggingface.co/models?search=Smaug)
- [x] [Poro 34B](https://huggingface.co/LumiOpen/Poro-34B)
- [x] [Bitnet b1.58 models](https://huggingface.co/1bitLLM)
- [x] [Flan T5](https://huggingface.co/models?search=flan-t5)
- [x] [Open Elm models](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca)
- [x] [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b) + [ChatGLM4-9b](https://huggingface.co/THUDM/glm-4-9b) + [GLMEdge-1.5b](https://huggingface.co/THUDM/glm-edge-1.5b-chat) + [GLMEdge-4b](https://huggingface.co/THUDM/glm-edge-4b-chat)
- [x] [GLM-4-0414](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e)
- [x] [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
- [x] [EXAONE-3.0-7.8B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct)
- [x] [FalconMamba Models](https://huggingface.co/collections/tiiuae/falconmamba-7b-66b9a580324dd1598b0f6d4a)
- [x] [Jais](https://huggingface.co/inceptionai/jais-13b-chat)
- [x] [Bielik-11B-v2.3](https://huggingface.co/collections/speakleash/bielik-11b-v23-66ee813238d9b526a072408a)
- [x] [RWKV-7](https://huggingface.co/collections/shoumenchougou/rwkv7-gxx-gguf)
- [x] [RWKV-6](https://github.com/BlinkDL/RWKV-LM)
- [x] [QRWKV-6](https://huggingface.co/recursal/QRWKV6-32B-Instruct-Preview-v0.1)
- [x] [GigaChat-20B-A3B](https://huggingface.co/ai-sage/GigaChat-20B-A3B-instruct)
- [X] [Trillion-7B-preview](https://huggingface.co/trillionlabs/Trillion-7B-preview)
- [x] [Ling models](https://huggingface.co/collections/inclusionAI/ling-67c51c85b34a7ea0aba94c32)
- [x] [LFM2 models](https://huggingface.co/collections/LiquidAI/lfm2-686d721927015b2ad73eaa38)
- [x] [Hunyuan models](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)
- [x] [BailingMoeV2 (Ring/Ling 2.0) models](https://huggingface.co/collections/inclusionAI/ling-v2-68bf1dd2fc34c306c1fa6f86)

#### Multimodal

- [x] [LLaVA 1.5 models](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e), [LLaVA 1.6 models](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2)
- [x] [BakLLaVA](https://huggingface.co/models?search=SkunkworksAI/Bakllava)
- [x] [Obsidian](https://huggingface.co/NousResearch/Obsidian-3B-V0.5)
- [x] [ShareGPT4V](https://huggingface.co/models?search=Lin-Chen/ShareGPT4V)
- [x] [MobileVLM 1.7B/3B models](https://huggingface.co/models?search=mobileVLM)
- [x] [Yi-VL](https://huggingface.co/models?search=Yi-VL)
- [x] [Mini CPM](https://huggingface.co/models?search=MiniCPM)
- [x] [Moondream](https://huggingface.co/vikhyatk/moondream2)
- [x] [Bunny](https://github.com/BAAI-DCAI/Bunny)
- [x] [GLM-EDGE](https://huggingface.co/models?search=glm-edge)
- [x] [Qwen2-VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)
- [x] [LFM2-VL](https://huggingface.co/collections/LiquidAI/lfm2-vl-68963bbc84a610f7638d5ffa)

</details>

<details>
<summary>Bindings</summary>

- Python: [ddh0/easy-llama](https://github.com/ddh0/easy-llama)
- Python: [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Go: [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp)
- Node.js: [withcatai/node-llama-cpp](https://github.com/withcatai/node-llama-cpp)
- JS/TS (llama.cpp server client): [lgrammel/modelfusion](https://modelfusion.dev/integration/model-provider/llamacpp)
- JS/TS (Programmable Prompt Engine CLI): [offline-ai/cli](https://github.com/offline-ai/cli)
- JavaScript/Wasm (works in browser): [tangledgroup/llama-cpp-wasm](https://github.com/tangledgroup/llama-cpp-wasm)
- Typescript/Wasm (nicer API, available on npm): [ngxson/wllama](https://github.com/ngxson/wllama)
- Ruby: [yoshoku/llama_cpp.rb](https://github.com/yoshoku/llama_cpp.rb)
- Rust (more features): [edgenai/llama_cpp-rs](https://github.com/edgenai/llama_cpp-rs)
- Rust (nicer API): [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp)
- Rust (more direct bindings): [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs)
- Rust (automated build from crates.io): [ShelbyJenkins/llm_client](https://github.com/ShelbyJenkins/llm_client)
- C#/.NET: [SciSharp/LLamaSharp](https://github.com/SciSharp/LLamaSharp)
- C#/VB.NET (more features - community license): [LM-Kit.NET](https://docs.lm-kit.com/lm-kit-net/index.html)
- Scala 3: [donderom/llm4s](https://github.com/donderom/llm4s)
- Clojure: [phronmophobic/llama.clj](https://github.com/phronmophobic/llama.clj)
- React Native: [mybigday/llama.rn](https://github.com/mybigday/llama.rn)
- Java: [kherud/java-llama.cpp](https://github.com/kherud/java-llama.cpp)
- Java: [QuasarByte/llama-cpp-jna](https://github.com/QuasarByte/llama-cpp-jna)
- Zig: [deins/llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)
- Flutter/Dart: [netdur/llama_cpp_dart](https://github.com/netdur/llama_cpp_dart)
- Flutter: [xuegao-tzx/Fllama](https://github.com/xuegao-tzx/Fllama)
- PHP (API bindings and features built on top of llama.cpp): [distantmagic/resonance](https://github.com/distantmagic/resonance) [(more info)](https://github.com/ggml-org/llama.cpp/pull/6326)
- Guile Scheme: [guile_llama_cpp](https://savannah.nongnu.org/projects/guile-llama-cpp)
- Swift [srgtuszy/llama-cpp-swift](https://github.com/srgtuszy/llama-cpp-swift)
- Swift [ShenghaiWang/SwiftLlama](https://github.com/ShenghaiWang/SwiftLlama)
- Delphi [Embarcadero/llama-cpp-delphi](https://github.com/Embarcadero/llama-cpp-delphi)
- Go (no CGo needed): [hybridgroup/yzma](https://github.com/hybridgroup/yzma)
- Android: [llama.android](/examples/llama.android)

</details>

<details>
<summary>UIs</summary>

*(to have a project listed here, it should clearly state that it depends on `llama.cpp`)*

- [AI Sublime Text plugin](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) (MIT)
- [BonzAI App](https://apps.apple.com/us/app/bonzai-your-local-ai-agent/id6752847988) (proprietary)
- [cztomsik/ava](https://github.com/cztomsik/ava) (MIT)
- [Dot](https://github.com/alexpinel/Dot) (GPL)
- [eva](https://github.com/ylsdamxssjxxdd/eva) (MIT)
- [iohub/collama](https://github.com/iohub/coLLaMA) (Apache-2.0)
- [janhq/jan](https://github.com/janhq/jan) (AGPL)
- [johnbean393/Sidekick](https://github.com/johnbean393/Sidekick) (MIT)
- [KanTV](https://github.com/zhouwg/kantv?tab=readme-ov-file) (Apache-2.0)
- [KodiBot](https://github.com/firatkiral/kodibot) (GPL)
- [llama.vim](https://github.com/ggml-org/llama.vim) (MIT)
- [LARS](https://github.com/abgulati/LARS) (AGPL)
- [Llama Assistant](https://github.com/vietanhdev/llama-assistant) (GPL)
- [LlamaLib](https://github.com/undreamai/LlamaLib) (Apache-2.0)
- [LLMFarm](https://github.com/guinmoon/LLMFarm?tab=readme-ov-file) (MIT)
- [LLMUnity](https://github.com/undreamai/LLMUnity) (MIT)
- [LMStudio](https://lmstudio.ai/) (proprietary)
- [LocalAI](https://github.com/mudler/LocalAI) (MIT)
- [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp) (AGPL)
- [MindMac](https://mindmac.app) (proprietary)
- [MindWorkAI/AI-Studio](https://github.com/MindWorkAI/AI-Studio) (FSL-1.1-MIT)
- [Mobile-Artificial-Intelligence/maid](https://github.com/Mobile-Artificial-Intelligence/maid) (MIT)
- [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) (Apache-2.0)
- [nat/openplayground](https://github.com/nat/openplayground) (MIT)
- [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all) (MIT)
- [ollama/ollama](https://github.com/ollama/ollama) (MIT)
- [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) (AGPL)
- [PocketPal AI](https://github.com/a-ghorbani/pocketpal-ai) (MIT)
- [psugihara/FreeChat](https://github.com/psugihara/FreeChat) (MIT)
- [ptsochantaris/emeltal](https://github.com/ptsochantaris/emeltal) (MIT)
- [pythops/tenere](https://github.com/pythops/tenere) (AGPL)
- [ramalama](https://github.com/containers/ramalama) (MIT)
- [semperai/amica](https://github.com/semperai/amica) (MIT)
- [withcatai/catai](https://github.com/withcatai/catai) (MIT)
- [Autopen](https://github.com/blackhole89/autopen) (GPL)

</details>

<details>
<summary>Tools</summary>

- [akx/ggify](https://github.com/akx/ggify) – download PyTorch models from HuggingFace Hub and convert them to GGML
- [akx/ollama-dl](https://github.com/akx/ollama-dl) – download models from the Ollama library to be used directly with llama.cpp
- [crashr/gppm](https://github.com/crashr/gppm) – launch llama.cpp instances utilizing NVIDIA Tesla P40 or P100 GPUs with reduced idle power consumption
- [gpustack/gguf-parser](https://github.com/gpustack/gguf-parser-go/tree/main/cmd/gguf-parser) - review/check the GGUF file and estimate the memory usage
- [Styled Lines](https://marketplace.unity.com/packages/tools/generative-ai/styled-lines-llama-cpp-model-292902) (proprietary licensed, async wrapper of inference part for game development in Unity3d with pre-built Mobile and Web platform wrappers and a model example)
- [unslothai/unsloth](https://github.com/unslothai/unsloth) – 🦥 exports/saves fine-tuned and trained models to GGUF (Apache-2.0)

</details>

<details>
<summary>Infrastructure</summary>

- [Paddler](https://github.com/intentee/paddler) - Open-source LLMOps platform for hosting and scaling AI in your own infrastructure
- [GPUStack](https://github.com/gpustack/gpustack) - Manage GPU clusters for running LLMs
- [llama_cpp_canister](https://github.com/onicai/llama_cpp_canister) - llama.cpp as a smart contract on the Internet Computer, using WebAssembly
- [llama-swap](https://github.com/mostlygeek/llama-swap) - transparent proxy that adds automatic model switching with llama-server
- [Kalavai](https://github.com/kalavai-net/kalavai-client) - Crowdsource end to end LLM deployment at any scale
- [llmaz](https://github.com/InftyAI/llmaz) - ☸️ Easy, advanced inference platform for large language models on Kubernetes.
</details>

<details>
<summary>Games</summary>

- [Lucy's Labyrinth](https://github.com/MorganRO8/Lucys_Labyrinth) - A simple maze game where agents controlled by an AI model will try to trick you.

</details>


## Supported backends

| Backend | Target devices |
| --- | --- |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [SYCL](docs/backend/SYCL.md) | Intel and Nvidia GPU |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [WebGPU [In Progress]](docs/build.md#webgpu) | All |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [Hexagon [In Progress]](docs/backend/hexagon/README.md) | Snapdragon |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |

## Obtaining and quantizing models

The [Hugging Face](https://huggingface.co) platform hosts a [number of LLMs](https://huggingface.co/models?library=gguf&sort=trending) compatible with `llama.cpp`:

- [Trending](https://huggingface.co/models?library=gguf&sort=trending)
- [LLaMA](https://huggingface.co/models?sort=trending&search=llama+gguf)

You can either manually download the GGUF file or directly use any `llama.cpp`-compatible models from [Hugging Face](https://huggingface.co/) or other model hosting sites, such as [ModelScope](https://modelscope.cn/), by using this CLI argument: `-hf <user>/<model>[:quant]`. For example:

```sh
llama-cli -hf ggml-org/gemma-3-1b-it-GGUF
```

By default, the CLI would download from Hugging Face, you can switch to other options with the environment variable `MODEL_ENDPOINT`. For example, you may opt to downloading model checkpoints from ModelScope or other model sharing communities by setting the environment variable, e.g. `MODEL_ENDPOINT=https://www.modelscope.cn/`.

After downloading a model, use the CLI tools to run it locally - see below.

`llama.cpp` requires the model to be stored in the [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) file format. Models in other data formats can be converted to GGUF using the `convert_*.py` Python scripts in this repo.

The Hugging Face platform provides a variety of online tools for converting, quantizing and hosting models with `llama.cpp`:

- Use the [GGUF-my-repo space](https://huggingface.co/spaces/ggml-org/gguf-my-repo) to convert to GGUF format and quantize model weights to smaller sizes
- Use the [GGUF-my-LoRA space](https://huggingface.co/spaces/ggml-org/gguf-my-lora) to convert LoRA adapters to GGUF format (more info: https://github.com/ggml-org/llama.cpp/discussions/10123)
- Use the [GGUF-editor space](https://huggingface.co/spaces/CISCai/gguf-editor) to edit GGUF meta data in the browser (more info: https://github.com/ggml-org/llama.cpp/discussions/9268)
- Use the [Inference Endpoints](https://ui.endpoints.huggingface.co/) to directly host `llama.cpp` in the cloud (more info: https://github.com/ggml-org/llama.cpp/discussions/9669)

To learn more about model quantization, [read this documentation](tools/quantize/README.md)

## [`llama-cli`](tools/cli)

#### A CLI tool for accessing and experimenting with most of `llama.cpp`'s functionality.

- <details open>
    <summary>Run in conversation mode</summary>

    Models with a built-in chat template will automatically activate conversation mode. If this doesn't occur, you can manually enable it by adding `-cnv` and specifying a suitable chat template with `--chat-template NAME`

    ```bash
    llama-cli -m model.gguf

    # > hi, who are you?
    # Hi there! I'm your helpful assistant! I'm an AI-powered chatbot designed to assist and provide information to users like you. I'm here to help answer your questions, provide guidance, and offer support on a wide range of topics. I'm a friendly and knowledgeable AI, and I'm always happy to help with anything you need. What's on your mind, and how can I assist you today?
    #
    # > what is 1+1?
    # Easy peasy! The answer to 1+1 is... 2!
    ```

    </details>

- <details>
    <summary>Run in conversation mode with custom chat template</summary>

    ```bash
    # use the "chatml" template (use -h to see the list of supported templates)
    llama-cli -m model.gguf -cnv --chat-template chatml

    # use a custom template
    llama-cli -m model.gguf -cnv --in-prefix 'User: ' --reverse-prompt 'User:'
    ```

    </details>

- <details>
    <summary>Constrain the output with a custom grammar</summary>

    ```bash
    llama-cli -m model.gguf -n 256 --grammar-file grammars/json.gbnf -p 'Request: schedule a call at 8pm; Command:'

    # {"appointmentTime": "8pm", "appointmentDetails": "schedule a a call"}
    ```

    The [grammars/](grammars/) folder contains a handful of sample grammars. To write your own, check out the [GBNF Guide](grammars/README.md).

    For authoring more complex JSON grammars, check out https://grammar.intrinsiclabs.ai/

    </details>


## [`llama-server`](tools/server)

#### A lightweight, [OpenAI API](https://github.com/openai/openai-openapi) compatible, HTTP server for serving LLMs.

- <details open>
    <summary>Start a local HTTP server with default configuration on port 8080</summary>

    ```bash
    llama-server -m model.gguf --port 8080

    # Basic web UI can be accessed via browser: http://localhost:8080
    # Chat completion endpoint: http://localhost:8080/v1/chat/completions
    ```

    </details>

- <details>
    <summary>Support multiple-users and parallel decoding</summary>

    ```bash
    # up to 4 concurrent requests, each with 4096 max context
    llama-server -m model.gguf -c 16384 -np 4
    ```

    </details>

- <details>
    <summary>Enable speculative decoding</summary>

    ```bash
    # the draft.gguf model should be a small variant of the target model.gguf
    llama-server -m model.gguf -md draft.gguf
    ```

    </details>

- <details>
    <summary>Serve an embedding model</summary>

    ```bash
    # use the /embedding endpoint
    llama-server -m model.gguf --embedding --pooling cls -ub 8192
    ```

    </details>

- <details>
    <summary>Serve a reranking model</summary>

    ```bash
    # use the /reranking endpoint
    llama-server -m model.gguf --reranking
    ```

    </details>

- <details>
    <summary>Constrain all outputs with a grammar</summary>

    ```bash
    # custom grammar
    llama-server -m model.gguf --grammar-file grammar.gbnf

    # JSON
    llama-server -m model.gguf --grammar-file grammars/json.gbnf
    ```

    </details>


## [`llama-perplexity`](tools/perplexity)

#### A tool for measuring the [perplexity](tools/perplexity/README.md) [^1] (and other quality metrics) of a model over a given text.

- <details open>
    <summary>Measure the perplexity over a text file</summary>

    ```bash
    llama-perplexity -m model.gguf -f file.txt

    # [1]15.2701,[2]5.4007,[3]5.3073,[4]6.2965,[5]5.8940,[6]5.6096,[7]5.7942,[8]4.9297, ...
    # Final estimate: PPL = 5.4007 +/- 0.67339
    ```

    </details>

- <details>
    <summary>Measure KL divergence</summary>

    ```bash
    # TODO
    ```

    </details>

[^1]: [https://huggingface.co/docs/transformers/perplexity](https://huggingface.co/docs/transformers/perplexity)

## [`llama-bench`](tools/llama-bench)

#### Benchmark the performance of the inference for various parameters.

- <details open>
    <summary>Run default benchmark</summary>

    ```bash
    llama-bench -m model.gguf

    # Output:
    # | model               |       size |     params | backend    | threads |          test |                  t/s |
    # | ------------------- | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         pp512 |      5765.41 ± 20.55 |
    # | qwen2 1.5B Q4_0     | 885.97 MiB |     1.54 B | Metal,BLAS |      16 |         tg128 |        197.71 ± 0.81 |
    #
    # build: 3e0ba0e60 (4229)
    ```

    </details>

## [`llama-simple`](examples/simple)

#### A minimal example for implementing apps with `llama.cpp`. Useful for developers.

- <details>
    <summary>Basic text completion</summary>

    ```bash
    llama-simple -m model.gguf

    # Hello my name is Kaitlyn and I am a 16 year old girl. I am a junior in high school and I am currently taking a class called "The Art of
    ```

    </details>


## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- See [good first issues](https://github.com/ggml-org/llama.cpp/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for tasks suitable for first contributions
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information
- Make sure to read this: [Inference at the edge](https://github.com/ggml-org/llama.cpp/discussions/205)
- A bit of backstory for those who are interested: [Changelog podcast](https://changelog.com/podcast/532)

## Other documentation

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development documentation

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

#### Seminal papers and background on the models

If your issue is with model generation quality, then please at least scan the following links and papers to understand the limitations of LLaMA models. This is especially important when choosing an appropriate model size and appreciating both the significant and subtle differences between LLaMA models and ChatGPT:
- LLaMA:
    - [Introducing LLaMA: A foundational, 65-billion-parameter large language model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
    - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- GPT-3.5 / InstructGPT / ChatGPT:
    - [Aligning language models to follow instructions](https://openai.com/research/instruction-following)
    - [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## XCFramework
The XCFramework is a precompiled version of the library for iOS, visionOS, tvOS,
and macOS. It can be used in Swift projects without the need to compile the
library from source. For example:
```swift
// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyLlamaPackage",
    targets: [
        .executableTarget(
            name: "MyLlamaPackage",
            dependencies: [
                "LlamaFramework"
            ]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b5046/llama-b5046-xcframework.zip",
            checksum: "c19be78b5f00d8d29a25da41042cb7afa094cbf6280a225abe614b03b20029ab"
        )
    ]
)
```
The above example is using an intermediate build `b5046` of the library. This can be modified
to use a different version by changing the URL and checksum.

## Completions
Command-line completion is available for some environments.

#### Bash Completion
```bash
$ build/bin/llama-cli --completion-bash > ~/.llama-completion.bash
$ source ~/.llama-completion.bash
```
Optionally this can be added to your `.bashrc` or `.bash_profile` to load it
automatically. For example:
```console
$ echo "source ~/.llama-completion.bash" >> ~/.bashrc
```

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
