#include "cublas.h"
#include "cublas_v2.h"


template<typename T>
__global__ void add_fusedQKV_bias_transpose_kernel(T* qkv_buf,
                                                   const T* __restrict QKV,
                                                   const T* __restrict qkv_bias,
                                                   const int batch_size,
                                                   const int seq_len,
                                                   const int head_num,
                                                   const int size_per_head);

template<typename T, typename T_IN>
void invokeSoftMax(T* buffer,
                         const T_IN* buffer_src,
                         const int batch_size,
                         const int seq_len,
                         const int kv_len,
                         const int head_num,
                         const T scalar,
                         cudaStream_t stream);

template<typename T>
void invokeTransposeQKV(T* dst,
                        T* src,
                        const int batch_size,
                        const int seq_len,
                        const int head_num,
                        const int size_per_head,
                        cudaStream_t stream);

template<typename T>
void invokeTransposeQKV2(T* dst,
                        T* src,
                        const int batch_size,
                        const int seq_len,
                        const int head_num,
                        const int size_per_head,
                        cudaStream_t stream);

template<typename T>
void invokeAddBias(T* output, const T* bias, const int m, const int n, cudaStream_t stream);

template<typename T>
void invokeAddMat_Mask(T* output, 
                    const float* matA, 
                    const float* matB, 
                    const int* mask,
                    const int batch_size,
                    const int seq_len,
                    const int head_num,
                    const int size_per_head,
                    cudaStream_t stream);