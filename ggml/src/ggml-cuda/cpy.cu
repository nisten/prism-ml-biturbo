#include "cpy.cuh"
#include "dequantize.cuh"
#include "cpy-utils.cuh"
#include "../ggml-common.h"
#include "../ggml-turboq.h"
#if defined(GGML_USE_MUSA) && defined(GGML_MUSA_MUDNN_COPY)
#include "ggml-musa/mudnn.cuh"
#endif // GGML_USE_MUSA && GGML_MUSA_MUDNN_COPY

typedef void (*cpy_kernel_t)(const char * cx, char * cdst);

const int CUDA_CPY_TILE_DIM_2D = 32; // 2D tile dimension for transposed blocks
const int CUDA_CPY_BLOCK_NM = 8;     // block size of 3rd dimension if available
const int CUDA_CPY_BLOCK_ROWS = 8;   // block dimension for marching through rows

template <cpy_kernel_t cpy_1>
static __global__ void cpy_scalar(const char * cx, char * cdst, const int64_t ne,
                                  const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
                                  const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11,
                                  const int64_t nb12, const int64_t nb13) {
    const int64_t i = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    // determine indices i03/i13, i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
    // then combine those indices with the corresponding byte offsets to get the total offsets
    const int64_t i03 = i/(ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int64_t x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int64_t i13 = i/(ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13 * nb13;

    cpy_1(cx + x_offset, cdst + dst_offset);
}

template <typename T>
static __global__ void cpy_scalar_transpose(const char * cx, char * cdst, const int64_t ne,
                               const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
                               const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11,
                               const int64_t nb12, const int64_t nb13) {

    const T* src = reinterpret_cast<const T*>(cx);
    T* dst = reinterpret_cast<T*>(cdst);

    const int64_t nmat = ne / (ne00 * ne01);
    const int64_t n = ne00 * ne01;

    const int x = blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.x;
    const int y = blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.y;
    const int tx = blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.x;  // transpose block offset
    const int ty = blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.y;

    __shared__ float tile[CUDA_CPY_TILE_DIM_2D][CUDA_CPY_TILE_DIM_2D+1];

#pragma unroll
    for (int i = 0; i < CUDA_CPY_BLOCK_NM; ++i) {

        const unsigned int imat = blockIdx.z * CUDA_CPY_BLOCK_NM + i;
        if (imat >= nmat)
            break;

#pragma unroll
        for (int j = 0; j < CUDA_CPY_TILE_DIM_2D; j += CUDA_CPY_BLOCK_ROWS) {
            if(x < ne01 && y + j < ne00){
                const int row = threadIdx.y+j;
                const int col = threadIdx.x * sizeof(float)/sizeof(T);
                T *tile2 = reinterpret_cast<T*>(tile[row]);
                tile2[col] = src[imat*n + (y+j)*ne01 + x];
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < CUDA_CPY_TILE_DIM_2D; j += CUDA_CPY_BLOCK_ROWS) {
            if (ty + j < ne01 && tx < ne00) {
                const int col = (threadIdx.y+j)*sizeof(float)/sizeof(T);
                const T *tile2 = reinterpret_cast<const T*>(tile[threadIdx.x]);
                dst[imat*n + (ty+j)*ne00 + tx] = tile2[col];
            }
        }
    }

    GGML_UNUSED_VARS(ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11,
        nb12, nb13);
}

static __device__ void cpy_blck_q8_0_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *)(cdsti);

#pragma unroll
    for (int j = 0; j < QK8_0; j += 2) {
        float2 dq;
        dequantize_q8_0(cxi, 0, j, dq);
        *(cdstf + j) = dq.x;
        *(cdstf + j + 1) = dq.y;
    }
}

template<dequantize_kernel_t dequant, int qk>
static __device__ void cpy_blck_q_f32(const char * cxi, char * cdsti) {
    float * cdstf = (float *)(cdsti);

#pragma unroll
    for (int j = 0; j < qk/2; j++) {
        float2 dq;
        dequant(cxi, 0, j, dq);
        *(cdstf + j) = dq.x;
        *(cdstf + j + qk/2) = dq.y;
    }
}

template <cpy_kernel_t cpy_blck, int qk>
static __global__ void cpy_f32_q(const char * cx, char * cdst, const int64_t ne,
                                 const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
                                 const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11,
                                 const int64_t nb12, const int64_t nb13) {
    const int64_t i = ((int64_t)blockDim.x*blockIdx.x + threadIdx.x)*qk;

    if (i >= ne) {
        return;
    }

    const int64_t i03 = i/(ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int64_t x_offset = i00*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int64_t i13 = i/(ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int64_t dst_offset = (i10/qk)*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template <cpy_kernel_t cpy_blck, int qk>
static __global__ void cpy_q_f32(const char * cx, char * cdst, const int64_t ne,
                                 const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
                                 const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11,
                                 const int64_t nb12, const int64_t nb13) {
    const int64_t i = ((int64_t)blockDim.x*blockIdx.x + threadIdx.x)*qk;

    if (i >= ne) {
        return;
    }

    const int64_t i03 = i/(ne00 * ne01 * ne02);
    const int64_t i02 = (i - i03*ne00*ne01*ne02 )/ (ne00*ne01);
    const int64_t i01 = (i - i03*ne00*ne01*ne02  -  i02*ne01*ne00) / ne00;
    const int64_t i00 = i - i03*ne00*ne01*ne02 - i02*ne01*ne00 - i01*ne00;
    const int64_t x_offset = (i00/qk)*nb00 + i01*nb01 + i02*nb02 + i03 * nb03;

    const int64_t i13 = i/(ne10 * ne11 * ne12);
    const int64_t i12 = (i - i13*ne10*ne11*ne12) / (ne10*ne11);
    const int64_t i11 = (i - i13*ne10*ne11*ne12 - i12*ne10*ne11) / ne10;
    const int64_t i10 = i - i13*ne10*ne11*ne12 - i12*ne10*ne11 - i11*ne10;
    const int64_t dst_offset = i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13;

    cpy_blck(cx + x_offset, cdst + dst_offset);
}

template<typename src_t, typename dst_t>
static __global__ void cpy_scalar_contiguous(const char * cx, char * cdst, const int64_t ne) {
    const int64_t i = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    const src_t * x = (const src_t *) cx;
    dst_t *     dst = (dst_t *) cdst;

    dst[i] = ggml_cuda_cast<dst_t>(x[i]);
}

template<typename src_t, typename dst_t>
static void ggml_cpy_scalar_contiguous_cuda(
    const char * cx, char * cdst, const int64_t ne,
cudaStream_t stream) {

    const int64_t num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_scalar_contiguous<src_t, dst_t><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
        (cx, cdst, ne);
}

template<typename src_t, typename dst_t, bool transposed = false>
static void ggml_cpy_scalar_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    if (transposed) {
        GGML_ASSERT(ne == ne00*ne01*ne02);  // ne[3] is 1 assumed
        int64_t ne00n, ne01n, ne02n;
        if (nb00 <= nb02) { // most likely safe to handle nb00 = nb02 case here
            ne00n = ne00;
            ne01n = ne01;
            ne02n = ne02;
        } else {
            ne00n = ne00;
            ne01n = ne01*ne02;
            ne02n = 1;
        }

        int64_t grid_x = (ne01n + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D;
        int64_t grid_y = (ne00n + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D;
        int64_t grid_z = (ne/(ne01n*ne00n) + CUDA_CPY_BLOCK_NM - 1) / CUDA_CPY_BLOCK_NM;
        GGML_ASSERT(grid_x < UINT_MAX);
        GGML_ASSERT(grid_y < USHRT_MAX);
        GGML_ASSERT(grid_z < USHRT_MAX);
        dim3 dimGrid(grid_x, grid_y, grid_z);
        dim3 dimBlock(CUDA_CPY_TILE_DIM_2D, CUDA_CPY_BLOCK_ROWS, 1);
        cpy_scalar_transpose<dst_t><<<dimGrid, dimBlock, 0, stream>>>
            (cx, cdst, ne, ne00n, ne01n, ne02n, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
    } else {
        const int64_t num_blocks = (ne + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
        GGML_ASSERT(num_blocks < UINT_MAX);
        cpy_scalar<cpy_1_scalar<src_t, dst_t>><<<num_blocks, CUDA_CPY_BLOCK_SIZE, 0, stream>>>
            (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
    }
}

static void ggml_cpy_f32_q8_0_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK8_0 == 0);
    const int64_t num_blocks = ne / QK8_0;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_q8_0, QK8_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_q8_0_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    const int64_t num_blocks = ne;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_q8_0_f32, QK8_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q4_0_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_0 == 0);
    const int64_t num_blocks = ne / QK4_0;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_q4_0, QK4_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_q4_0_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    const int64_t num_blocks = ne;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q4_1_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_1 == 0);
    const int64_t num_blocks = ne / QK4_1;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_q4_1, QK4_1><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_q4_1_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    const int64_t num_blocks = ne;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
         ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q5_0_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK5_0 == 0);
    const int64_t num_blocks = ne / QK5_0;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_q5_0, QK5_0><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_q5_0_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    const int64_t num_blocks = ne;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_q5_1_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK5_1 == 0);
    const int64_t num_blocks = ne / QK5_1;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_q5_1, QK5_1><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_q5_1_f32_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream) {
    const int64_t num_blocks = ne;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_q_f32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1><<<num_blocks, 1, 0, stream>>>(
        cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

static void ggml_cpy_f32_iq4_nl_cuda(
    const char * cx, char * cdst, const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t nb00, const int64_t nb01, const int64_t nb02,
    const int64_t nb03, const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13, cudaStream_t stream) {

    GGML_ASSERT(ne % QK4_NL == 0);
    const int64_t num_blocks = ne / QK4_NL;
    GGML_ASSERT(num_blocks < UINT_MAX);
    cpy_f32_q<cpy_blck_f32_iq4_nl, QK4_NL><<<num_blocks, 1, 0, stream>>>
        (cx, cdst, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13);
}

// ---------------------------------------------------------------------------
// TBQ4_0 → F32 dequantize on CUDA
//
// block_tbq4_0: uint8_t qs[128] (nibble-packed, 256 values), ggml_half d
// Dequant: nibble → codebook*scale_down → Q^T matvec (inverse rotation) → * norm
// The 256-element block uses two independent 128-element sub-blocks with the same Q.
// ---------------------------------------------------------------------------

#define TBQ4_DIM 128
#define TBQ4_QK  256

// Codebook × scale_down precomputed (scale_down = 1/sqrt(256) = 1/16)
__device__ __constant__ float d_tbq4_codebook_scaled[16];

// Q^T stored row-major on device: d_tbq4_QT[j*128 + i] = Q^T_{ji} = Q_{ij}
// Allows thread j to read its row contiguously for the dot product.
static float * g_d_tbq4_QT = nullptr;
static bool    g_tbq4_initialized = false;

static void tbq4_cuda_init(void) {
    if (g_tbq4_initialized) return;

    const int64_t d = TBQ4_DIM;
    const uint64_t seed = turboq_seed_from_row(0);
    const float scale_down = 1.0f / sqrtf((float)TBQ4_QK);

    // Get Q column-major from CPU turboq
    float * Q_cm = (float *)malloc((size_t)(d * d) * sizeof(float));
    turboq_get_rotation_matrix(d, seed, Q_cm);

    // Build Q^T row-major: QT_rm[j*d + i] = Q_cm[i + j*d]
    float * QT_rm = (float *)malloc((size_t)(d * d) * sizeof(float));
    for (int64_t j = 0; j < d; j++) {
        for (int64_t i = 0; i < d; i++) {
            QT_rm[j * d + i] = Q_cm[i + j * d];
        }
    }
    free(Q_cm);

    CUDA_CHECK(cudaMalloc(&g_d_tbq4_QT, (size_t)(d * d) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(g_d_tbq4_QT, QT_rm, (size_t)(d * d) * sizeof(float), cudaMemcpyHostToDevice));
    free(QT_rm);

    // Upload scaled codebook to constant memory
    float cb_scaled[16];
    // turboq_codebook_4bit values (not accessible here directly, hardcode same values)
    static const float cb[16] = {
        -2.7326f, -2.0690f, -1.6180f, -1.2562f,
        -0.9424f, -0.6568f, -0.3881f, -0.1284f,
         0.1284f,  0.3881f,  0.6568f,  0.9424f,
         1.2562f,  1.6180f,  2.0690f,  2.7326f
    };
    for (int i = 0; i < 16; i++) {
        cb_scaled[i] = cb[i] * scale_down;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_tbq4_codebook_scaled, cb_scaled, sizeof(cb_scaled)));

    g_tbq4_initialized = true;
}

// Each CUDA block handles one TBQ4_0 block (256 output elements).
// 256 threads: thread t computes output[t].
__global__ void cpy_tbq4_0_f32_kernel(
    const char * __restrict__ src_blocks,
    float * __restrict__ dst,
    int64_t nb,
    const float * __restrict__ d_QT  // Q^T row-major [128×128] on device
) {
    const int64_t blk_idx = (int64_t)blockIdx.x;
    if (blk_idx >= nb) return;

    const int tid = threadIdx.x;  // 0..255

    // block_tbq4_0 layout: qs[128], d[2]
    const int64_t block_bytes = 128 + 2;
    const uint8_t * qs = (const uint8_t *)(src_blocks + blk_idx * block_bytes);
    // d is at offset 128
    const ggml_half d_fp16 = *reinterpret_cast<const ggml_half *>(src_blocks + blk_idx * block_bytes + 128);
    const float norm = __half2float(d_fp16);

    // Shared: rotated vector (codebook*scale_down, before inverse rotation)
    __shared__ float rotated[TBQ4_QK];

    // Step 1: unpack nibble → codebook value × scale_down
    {
        const uint8_t byte = qs[tid / 2];
        const uint8_t idx  = (tid % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        rotated[tid] = d_tbq4_codebook_scaled[idx];
    }
    __syncthreads();

    // Step 2: inverse rotation for each 128-element sub-block.
    // output[tid] = (Q^T * rotated_sub)[tid % 128] * norm
    // where rotated_sub = rotated[sub_block * 128 : sub_block * 128 + 128]
    //
    // Q^T row-major: row local_tid is contiguous at g_d_tbq4_QT + local_tid*128
    const int local_tid = tid % TBQ4_DIM;    // 0..127
    const int sub_block = tid / TBQ4_DIM;    // 0 or 1
    const float * rot_sub  = rotated + sub_block * TBQ4_DIM;
    const float * qt_row   = d_QT + local_tid * TBQ4_DIM;

    float sum = 0.0f;
    // Unroll hint: 128 iterations
    #pragma unroll 8
    for (int i = 0; i < TBQ4_DIM; i++) {
        sum += qt_row[i] * rot_sub[i];
    }

    dst[blk_idx * TBQ4_QK + tid] = sum * norm;
}

static void ggml_cpy_tbq4_0_f32_cuda(
    const char * cx, char * cdst,
    const int64_t ne,
    const int64_t ne00, const int64_t ne01, const int64_t ne02,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    cudaStream_t stream
) {
    GGML_ASSERT(ne % TBQ4_QK == 0);

    tbq4_cuda_init();

    const int64_t nb = ne / TBQ4_QK;
    GGML_ASSERT(nb < (int64_t)INT_MAX);

    cpy_tbq4_0_f32_kernel<<<(unsigned int)nb, TBQ4_QK, 0, stream>>>(cx, (float *)cdst, nb, g_d_tbq4_QT);

    (void)ne00; (void)ne01; (void)ne02; (void)nb00; (void)nb01; (void)nb02; (void)nb03;
    (void)ne10; (void)ne11; (void)ne12; (void)nb10; (void)nb11; (void)nb12; (void)nb13;
}

void ggml_cuda_cpy(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, ggml_tensor * src1) {
    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne == ggml_nelements(src1));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    //GGML_ASSERT(src0->ne[3] == 1);

    const int64_t nb00 = src0->nb[0];
    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];
    const int64_t nb03 = src0->nb[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];

    //GGML_ASSERT(src1->ne[3] == 1);

    const int64_t nb10 = src1->nb[0];
    const int64_t nb11 = src1->nb[1];
    const int64_t nb12 = src1->nb[2];
    const int64_t nb13 = src1->nb[3];

    cudaStream_t main_stream = ctx.stream();

    char * src0_ddc = (char *) src0->data;
    char * src1_ddc = (char *) src1->data;

    const bool contiguous_srcs = ggml_is_contiguous(src0) && ggml_is_contiguous(src1);
    const bool can_be_transposed = nb01 == (int64_t)ggml_element_size(src0) &&
        src0->ne[3] == 1 && nb02 == ne00 * ne01 * (int64_t)ggml_element_size(src0);

    if (src0->type == src1->type && contiguous_srcs) {
        GGML_ASSERT(ggml_nbytes(src0) == ggml_nbytes(src1));
#if defined(GGML_USE_MUSA) && defined(GGML_MUSA_MUDNN_COPY)
        if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16) {
            CUDA_CHECK(mudnnMemcpyAsync(ctx, src1, src0));
        } else
#endif // GGML_USE_MUSA && GGML_MUSA_MUDNN_COPY
        {
            CUDA_CHECK(cudaMemcpyAsync(src1_ddc, src0_ddc, ggml_nbytes(src0), cudaMemcpyDeviceToDevice, main_stream));
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        if (can_be_transposed) {
            ggml_cpy_scalar_cuda<float, float, true>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        } else {
            ggml_cpy_scalar_cuda<float, float>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<float, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<float, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<float, half>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<float, half>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
        ggml_cpy_f32_q8_0_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q8_0_f32_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_0) {
        ggml_cpy_f32_q4_0_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_0_f32_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q4_1) {
        ggml_cpy_f32_q4_1_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q4_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q4_1_f32_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_0) {
        ggml_cpy_f32_q5_0_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_0_f32_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_IQ4_NL) {
        ggml_cpy_f32_iq4_nl_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q5_1) {
        ggml_cpy_f32_q5_1_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_Q5_1 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_q5_1_f32_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_TBQ4_0 && src1->type == GGML_TYPE_F32) {
        ggml_cpy_tbq4_0_f32_cuda
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16) {
        if (can_be_transposed) {
            ggml_cpy_scalar_cuda<half, half, true>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        } else {
            ggml_cpy_scalar_cuda<half, half>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_BF16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<half, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<half, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<half, float>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<half, float>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16) {
        if (can_be_transposed) {
            ggml_cpy_scalar_cuda<nv_bfloat16, nv_bfloat16, true>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        } else {
            ggml_cpy_scalar_cuda<nv_bfloat16, nv_bfloat16>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F16) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<nv_bfloat16, half>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<nv_bfloat16, half>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<nv_bfloat16, float>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<nv_bfloat16, float>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32) {
        if (can_be_transposed) {
            ggml_cpy_scalar_cuda<int32_t, int32_t, true>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        } else {
            ggml_cpy_scalar_cuda<int32_t, int32_t>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<float, int32_t>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<float, int32_t>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else if (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_F32) {
        if (contiguous_srcs) {
            ggml_cpy_scalar_contiguous_cuda<int32_t, float>
                (src0_ddc, src1_ddc, ne, main_stream);
        } else {
            ggml_cpy_scalar_cuda<int32_t, float>
                (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb13, main_stream);
        }
    } else {
        GGML_ABORT("%s: unsupported type combination (%s to %s)\n", __func__,
                ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

void ggml_cuda_dup(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    ggml_cuda_cpy(ctx, src0, dst);
}
