# KANBAN.md — prism-ml-biturbo Project Guide

## Project Summary

Extremely alpha fork of PrismML's llama.cpp (Q1_0_g128 1-bit inference) adding:
1. dp4a kernel for Q1_0_g128 on non-RTX Turing GPUs (GTX 1660 Ti)
2. TBQ4_0 (TurboQuant 4-bit) KV cache with full CUDA quantize + dequantize
3. FWHT (Fast Walsh-Hadamard Transform) replacing Householder for GPU rotation

Status: works on one GPU (GTX 1660 Ti, sm_75). Untested elsewhere.

---

## Critical Safety Rules

### 1. NEVER omit -DCMAKE_CUDA_ARCHITECTURES=XX

CUDA 13.x default arch auto-detection produces BROKEN PTX = silent garbage output.

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
```

### 2. CPU and CUDA use DIFFERENT rotations — DO NOT MIX

- CPU path (ggml-turboq.c): Householder QR rotation (O(n^2))
- CUDA path (tbq-wht.cuh): FWHT rotation (O(n log n))

These are DIFFERENT orthogonal transforms. A TBQ4_0 block quantized by one
CANNOT be dequantized by the other. Result: silent garbage.

- Full GPU mode (no -nkvo): CUDA quant + CUDA dequant = CORRECT
- `-nkvo` with TBQ4_0: CPU quant + CUDA dequant = BROKEN
- Full CPU mode: CPU quant + CPU dequant = consistent but broken for other reasons

### 3. TBQ3_0 has NO CUDA support — only TBQ4_0 works on GPU

### 4. The FWHT seed is FROZEN FOREVER

```c
uint64_t state = 0x517cc1b727220a95ULL;  // DO NOT CHANGE
```

Changing this seed, the splitmix64 algorithm, or sign generation order
makes ALL quantized data garbage.

### 5. CPU-only inference with TBQ4_0 produces garbage (undiagnosed)

---

## Architecture

```
                      PrismML's llama.cpp fork
                             |
                    +--------+--------+
                    |                 |
              Q1_0_g128          Standard quants
              (1-bit weights)    (Q4_0, Q8_0, etc.)
                    |
            +-------+-------+
            |               |
        MMA path        dp4a path (OUR FIX)
        (tensor cores)  (non-RTX Turing)
                    |
            +-------+-------+
            |               |
        f16 KV cache    TBQ4_0 KV cache (OUR ADDITION)
        (baseline)      (3.94x compression)
                            |
                    +-------+-------+
                    |               |
              SET_ROWS          CPY
              (F32->TBQ4_0)    (TBQ4_0->F32)
              FWHT forward     FWHT inverse
              cpy-utils.cuh    cpy.cu
```

---

## File Map

### New files we added
| File | Purpose |
|------|---------|
| `ggml/src/ggml-cuda/tbq-wht.cuh` | FWHT constants, thread-safe init, quantize helper |
| `ggml/src/ggml-turboq.c` | CPU rotation + quantize/dequantize (688 lines, Householder QR) |
| `ggml/src/ggml-turboq.h` | Rotation API header |
| `ggml/src/ggml-turboq-tables.h` | Lloyd-Max codebooks (2/3/4-bit) |

### Modified CUDA files
| File | Change |
|------|--------|
| `ggml/src/ggml-cuda/cpy.cu` | FWHT inverse dequant kernel (cpy_tbq4_0_f32_kernel) |
| `ggml/src/ggml-cuda/cpy-utils.cuh` | FWHT forward quant (quantize_f32_tbq4_0_block) |
| `ggml/src/ggml-cuda/set-rows.cu` | TBQ4_0 dispatch |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | TBQ4_0 in SET_ROWS + CPY supports_op |
| `ggml/src/ggml-cuda/mmq.cuh` | dp4a for Q1_0_g128 |

### Modified core/CPU/llama files
| File | Change |
|------|--------|
| `ggml/include/ggml.h` | Type enums TBQ3_0=42, TBQ4_0=43 |
| `ggml/src/ggml-common.h` | block_tbq3_0 (98B), block_tbq4_0 (130B) |
| `ggml/src/ggml.c` | type_traits, ftype mapping |
| `ggml/src/ggml-cpu/quants.c` | vec_dot for TBQ3_0/TBQ4_0 |
| `src/llama-kv-cache.cpp` | 3D view for TBQ K/V |
| `src/llama-graph.cpp` | TBQ cast+reshape before attention |
| `src/llama-context.cpp` | TBQ KV cache init |
| `common/arg.cpp` | -ctk tbq4_0 / -ctv tbq4_0 CLI |

---

## Performance

| Config | Gen t/s | Notes |
|--------|---------|-------|
| f16 KV baseline | 53.3 | Full GPU, no compression |
| TBQ4_0 full GPU | 34.4 | 3.94x KV compression, ~35% overhead |
| TBQ4_0 -nkvo | 9.7 | PCIe bottleneck, DO NOT USE |

35% overhead comes from: FWHT rotation (7 butterfly passes), single-threaded
quantize in SET_ROWS template, Lloyd-Max boundary search (15 comparisons/value).

---

## Known Bugs and Landmines

1. CPU/CUDA rotation mismatch (Householder vs FWHT) — silent garbage if mixed
2. CPU-only TBQ4_0 = garbage output at 0.3 t/s (undiagnosed)
3. TBQ3_0 no GPU path — crashes or silent CPU fallback
4. CUDA 13.x arch auto-detect — garbage without -DCMAKE_CUDA_ARCHITECTURES
5. Thread-local memory leaks in ggml-turboq.c — six tl_* buffers never freed
6. Codebook values duplicated between tbq-wht.cuh and ggml-turboq-tables.h
7. Single-GPU only — tbq_wht_init() inits constants on current device only
8. turboq_seed_from_row ignores row_idx — returns constant (correct but misleading)

---

## Reference Repos

| Repo | What to learn from it |
|------|----------------------|
| [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) | Full CUDA TBQ with FWHT, custom multi-threaded SET_ROWS kernel, 99.6% prefill speed |
| [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) | Metal+CUDA+HIP, Householder rotation, broader platform support |
| [elusznik PR #21089](https://github.com/ggml-org/llama.cpp/pull/21089) | CPU-only TBQ3_0/TBQ4_0, our CPU path based on this |
| [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) | Our upstream base, Q1_0_g128 1-bit kernels |

---

## TODO

### High priority
- [ ] Port FWHT to CPU path (replace Householder) — fixes CPU/CUDA mismatch
- [ ] Benchmark 16K context with full GPU KV (measure VRAM)
- [ ] Test max context length in 6 GB VRAM
- [ ] Multi-threaded CUDA quantize for SET_ROWS (prefill bottleneck)

### Medium priority
- [ ] Benchmark on RTX 3060, 4060, A100
- [ ] TBQ3_0 CUDA implementation
- [ ] Deduplicate codebook values (tbq-wht.cuh should include turboq-tables.h)
- [ ] Warp-level FWHT for first 5 butterfly passes (h <= 32, no __syncthreads needed)
- [ ] Multi-GPU constant memory init

### Low priority
- [ ] Upstream dp4a Q1_0_g128 fix to PrismML
- [ ] Fix CPU-only TBQ4_0 inference
- [ ] Free thread-local buffers in ggml-turboq.c

---

## Testing

```bash
# Build
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --config Release -j$(nproc)

# Quick sanity (should produce coherent answer, ~34 t/s gen)
./build/bin/llama-cli -m /path/to/Bonsai-8B.gguf \
  -c 512 -ngl 99 -fa on -ctk tbq4_0 -ctv tbq4_0 \
  -p "What is 2+2?" -n 20 --temp 0.1

# Baseline (should be ~53 t/s gen)
./build/bin/llama-cli -m /path/to/Bonsai-8B.gguf \
  -c 512 -ngl 99 -fa on \
  -p "What is 2+2?" -n 20 --temp 0.1
```

If output is garbage or < 5 t/s:
1. Did you use -DCMAKE_CUDA_ARCHITECTURES=XX?
2. Did you use -nkvo with TBQ4_0? (broken — CPU/CUDA rotation mismatch)
3. Is the model Q1_0_g128? (standard models won't work with PrismML kernels)
