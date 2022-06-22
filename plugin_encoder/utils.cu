#include "cublas.h"
#include "cublas_v2.h"
#include <string>
#include <vector>
#include <cassert>

int next_pow2(int a)
{
    int rval = 32;
    if (a > 32) {
        while (rval < a) {
            rval <<= 1;
        }
    }
    return rval;
}

// template<typename T>
// __global__ void calAttnScore(T* attn_score,
//                                       const T* ac,
//                                       const T* bd,
//                                       const T* attn_mask,
//                                       const int off0,
//                                       const int off1,
//                                       const int seq_len,
//                                       const float p)
// {
//     int batch = blockIdx.x;
//     int head = blockIdx.y;
//     int seq1 = blockIdx.z;
//     int seq2 = threadIdx.x;

//     int offset = batch * off0 + head * off1 + seq1 * seq_len;
//     int index = offset + seq2;
//     int out_index;
//     T score;
//     T mask;
//     T large_value = -1e4;
//     if (sizeof(T) == 4) {
//         large_value = -1e30f;
//     }
//     if (seq2 < seq_len) {
//         score = ac[index] + bd[index];
//         score = score * p;

//         out_index = batch * off1 + seq1 * seq_len + seq2;
//         mask = attn_mask[out_index] * (large_value);
//         score = score + mask;
//     }
//     // softmax(attn_score+offset,seq_len, seq2);
//     __shared__ float s_sum, s_max;
//     float tmp = seq2 < seq_len ? score : large_value;
//     float max_val = blockReduceMax<float>(tmp);
//     if (seq2 == 0) {
//         s_max = max_val;
//     }
//     __syncthreads();
//     float qk_tmp = seq2 < seq_len ? __expf((float)(tmp - s_max)) : 0.0f;
//     float sum_val = blockReduceSum<float>(qk_tmp);
//     __syncthreads();
//     if (seq2 == 0) {
//         s_sum = sum_val;
//     }
//     __syncthreads();
//     if (seq2 < seq_len) {
//         attn_score[index] = (T)(qk_tmp / s_sum);
//     }
//     // end softmax

//     // offset = seq2;
//     // while (offset < voff2) {
//     //     out_index = batch * voff0 + head * v_o_off1 + seq1 * voff2 + offset;
//     //     index = batch * voff0 + seq1 * v_i_off1 + head * voff2 + offset;
//     //     value_buf_trans[out_index] = value_buf[index];
//     //     offset += seq_len;
//     // }
// }

template<typename T>
inline __device__ T ldg(const T* val) {
    return __ldg(val);
}

template<typename T>
__global__ void add_fusedQKV_bias_transpose_kernel(T* qkv_buf,
                                                   const T* __restrict QKV,
                                                   const T* __restrict qkv_bias,
                                                   const int batch_size,
                                                   const int seq_len,
                                                   const int head_num,
                                                   const int size_per_head)
{

    const int n = head_num * size_per_head;
    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < batch_size * seq_len * n;
         index += gridDim.x * blockDim.x) {
        int bias_id = index % (n);
        T val = ldg(&QKV[index]) + ldg(&qkv_bias[bias_id]);

        int tmp_index = index;
        const int target_batch_id = tmp_index / (seq_len * n);
        tmp_index -= target_batch_id * seq_len * n;     
        const int seq_id = tmp_index / (n);             
        tmp_index -= seq_id * n;                  
        const int head_id = tmp_index / size_per_head;
        const int size_id = tmp_index - head_id * size_per_head;

        qkv_buf[target_batch_id * head_num * seq_len * size_per_head 
                + head_id * seq_len * size_per_head
                + seq_id * size_per_head + size_id] = val;
    }
}

template __global__ void add_fusedQKV_bias_transpose_kernel(float* qkv_buf,
                                            const float* __restrict QKV,
                                            const float* __restrict qkv_bias,
                                            const int batch_size,
                                            const int seq_len,
                                            const int head_num,
                                            const int size_per_head);


template<typename T>
__global__ void add_bias_kernel(T* qkv_buf,
                                const T* __restrict QKV,
                                const T* __restrict qkv_bias,
                                const int batch_size,
                                const int seq_len,
                                const int n)
{
    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < batch_size * seq_len * n;
         index += gridDim.x * blockDim.x) {
        int bias_id = index % (n);
        T val = ldg(&QKV[index]) + ldg(&qkv_bias[bias_id]);
        qkv_buf[index] = val;
    }
}



#define FINAL_MASK 0xffffffff

template<typename T>
__inline__ __device__ T warpReduceSum(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    return val;
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceSum(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T>(val);

    return val;
}

template<typename T>
__inline__ __device__ T warpReduceMax(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
    return val;
}

/* Calculate the maximum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceMax(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;  // in-warp idx
    int wid = threadIdx.x >> 5;     // warp idx

    val = warpReduceMax(val);  // get maxx in each warp

    if (lane == 0)  // record in-warp maxx by warp Idx
        shared[wid] = val;

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : -1e20f;
    val = warpReduceMax(val);

    return val;
}


template<int ITEMS_PER_THREAD, typename T, typename T_IN>
__global__ void softmax_kernel_v5(T* qk_buf_,
                                  const T_IN* qk_buf_src,
                                  const int batch_size,
                                  const int head_num,
                                  const int seq_len,
                                  const int kv_len,
                                  const T scalar)
{
    for (int seq_id = blockIdx.x; seq_id < seq_len; seq_id += gridDim.x) {
        float data[ITEMS_PER_THREAD];
        int qk_offset;
        __shared__ float s_mean, s_max;
        float local_max = -1e20f;

        for (int i = 0; blockDim.x * i + threadIdx.x < kv_len; i++) {
            qk_offset =
                ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * kv_len + blockDim.x * i + threadIdx.x;

            float qk = static_cast<float>(qk_buf_src[qk_offset]);

            data[i] = qk * static_cast<float>(scalar);
            local_max = fmax(local_max, data[i]);
        }

        float max_val = blockDim.x <= 32 ? warpReduceMax(local_max) : blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        for (int i = 0; blockDim.x * i + threadIdx.x < kv_len; i++) {
            data[i] = __expf(data[i] - s_max);
            local_sum += data[i];
        }
        float sum_val = blockDim.x <= 32 ? warpReduceSum(local_sum) : blockReduceSum<float>(local_sum);
        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < kv_len; i++) {
            qk_offset =
                ((blockIdx.y * head_num + blockIdx.z) * seq_len + seq_id) * kv_len + blockDim.x * i + threadIdx.x;
            qk_buf_[qk_offset] = (T)(data[i] * s_mean);
        }
    }
}

#define SOFTMAX_KERNEL(ITEMS_PER_THREAD)                                                                               \
    block.x /= ITEMS_PER_THREAD;                                                                                       \
    assert(block.x <= 1024);                                                                                           \
    if (is_half2) {                                                                                                    \
    }                                                                                                                  \
    else {                                                                                                             \
        softmax_kernel_v5<ITEMS_PER_THREAD, T, T_IN>                                                                   \
            <<<grid, block, 0, stream>>>(buffer, buffer_src, batch_size, head_num, seq_len, kv_len, scalar);        \
    }


template<typename T, typename T_IN>
void invokeSoftMax(T* buffer,
                         const T_IN* buffer_src,
                         const int batch_size,
                         const int seq_len,
                         const int kv_len,
                         const int head_num,
                         const T scalar,
                         cudaStream_t stream)
{

    dim3 grid(seq_len, batch_size, head_num);
    if (batch_size * head_num > 360) {
        grid.x = ceil(float(seq_len) / 32.0f);
    }

    bool is_half2 = sizeof(T) == 2 && sizeof(T_IN) == 2 && kv_len % 2 == 0;
    dim3 block((kv_len / (is_half2 ? 2 : 1) + 31) / 32 * 32);

    if (block.x > 3072 && block.x <= 4096) {
        SOFTMAX_KERNEL(4)
    }
    if (block.x > 2048) {
        SOFTMAX_KERNEL(3)
    }
    else if (block.x > 1024) {
        SOFTMAX_KERNEL(2)
    }
    else if (block.x > 0) {
        SOFTMAX_KERNEL(1)
    }
    else {
        assert (1==2);
        // FT_CHECK(seq_len <= 4096);
    }
}

template void invokeSoftMax(float* buffer,
                         const float* buffer_src,
                         const int batch_size,
                         const int seq_len,
                         const int kv_len,
                         const int head_num,
                         const float scalar,
                         cudaStream_t stream);

template<typename T>
__global__ void
transpose(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int batch_id = blockIdx.x / (head_num * seq_len);
    int seq_id = blockIdx.x % seq_len;
    int head_id = (blockIdx.x % (head_num * seq_len)) / seq_len;
    dst[batch_id * (head_num * seq_len * size_per_head) + seq_id * head_num * size_per_head + head_id * size_per_head
        + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
}

template<typename T>
void invokeTransposeQKV(T* dst,
                        T* src,
                        const int batch_size,
                        const int seq_len,
                        const int head_num,
                        const int size_per_head,
                        cudaStream_t stream)
{
    dim3 grid, block;
    if (sizeof(T) == 2) {
        int seq_per_block = 1;
        grid.x = batch_size * head_num * seq_len / seq_per_block;
        while (seq_per_block < 4 && grid.x % 2 == 0) {
            grid.x /= 2;
            seq_per_block *= 2;
        }

        // FT_CHECK(grid.x * seq_per_block == batch_size * head_num * seq_len);

        if (seq_per_block * size_per_head % 2 == 0) {
            block.x = seq_per_block * size_per_head / 2;
            if (std::is_same<T, half>::value) {
                transpose<half2><<<grid, block, 0, stream>>>(
                    (half2*)src, (half2*)dst, batch_size, seq_len, head_num, size_per_head / 2);
            }

        }
        else {
            block.x = seq_per_block * size_per_head;
            transpose<T><<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head);
        }
    }
    else {
        const int seq_per_block = 1;
        grid.x = batch_size * head_num * seq_len / seq_per_block;
        block.x = seq_per_block * size_per_head;
        transpose<T><<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head);
    }
}
template
void invokeTransposeQKV(float* dst,
                        float* src,
                        const int batch_size,
                        const int seq_len,
                        const int head_num,
                        const int size_per_head,
                        cudaStream_t stream);

template<typename T>
__global__ void
transpose2(T* src, T* dst, const int batch_size, const int seq_len, const int head_num, const int size_per_head)
{
    int batch_id = blockIdx.x / (head_num * seq_len);
    int head_id = blockIdx.x % head_num;
    //int seq_id = blockIdx.x % seq_len;
    int seq_id = (blockIdx.x % (head_num * seq_len)) / head_num;
    for (int i = 0; i < batch_size; i++){
        //dst[i*seq_len*head_num*size_per_head + blockIdx.x * size_per_head + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
        dst[i*seq_len*head_num*size_per_head + batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head + seq_id * size_per_head
        + threadIdx.x] = src[blockIdx.x * size_per_head + threadIdx.x];
    }
    
}

template<typename T>
void invokeTransposeQKV2(T* dst,
                        T* src,
                        const int batch_size,
                        const int seq_len,
                        const int head_num,
                        const int size_per_head,
                        cudaStream_t stream)
{
    dim3 grid, block;
    if (sizeof(T) == 2) {
        int seq_per_block = 1;
        assert (3==4);
        grid.x = batch_size * head_num * seq_len / seq_per_block;
        while (seq_per_block < 4 && grid.x % 2 == 0) {
            grid.x /= 2;
            seq_per_block *= 2;
        }

        if (seq_per_block * size_per_head % 2 == 0) {
            block.x = seq_per_block * size_per_head / 2;
            if (std::is_same<T, half>::value) {
                transpose2<half2><<<grid, block, 0, stream>>>(
                    (half2*)src, (half2*)dst, batch_size, seq_len, head_num, size_per_head / 2);
            }

        }
        else {
            block.x = seq_per_block * size_per_head;
            transpose2<T><<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head);
        }
    }
    else {
        const int seq_per_block = 1;
        grid.x = 1 * head_num * seq_len / seq_per_block;
        // grid.x = batch_size * head_num * seq_len / seq_per_block;
        block.x = seq_per_block * size_per_head;
        transpose2<T><<<grid, block, 0, stream>>>(src, dst, batch_size, seq_len, head_num, size_per_head);
    }
}

template
void invokeTransposeQKV2(float* dst,
                        float* src,
                        const int batch_size,
                        const int seq_len,
                        const int head_num,
                        const int size_per_head,
                        cudaStream_t stream);

template<typename T>
__global__ void addBias(T* output, const T* bias, const int m, const int n)
{
    const int col_index = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_index < n) {
        T bias_val = (bias == nullptr) ? (T)(0.0f) : bias[col_index];
        output[blockIdx.x * n + col_index] = output[blockIdx.x * n + col_index] + bias_val;
    }
}

template<typename T>
void invokeAddBias(T* output, const T* bias, const int m, const int n, cudaStream_t stream)
{
    int blocks_per_row = ceil(float(n) / 1024);
    dim3 grid(m, blocks_per_row);
    dim3 block(min(n, 1024));
    addBias<<<grid, block, 0, stream>>>(output, bias, m, n);
}
template void invokeAddBias(float* output, const float* bias, const int m, const int n, cudaStream_t stream);


template<typename T>
__global__ void addMat(T* dst, const T* ac, const T* bd, const int* mask, const int off0, const int off1, const int time2)
{
    
    int batch = blockIdx.x;
    int head = blockIdx.y;
    int seq1 = blockIdx.z;
    int seq2 = threadIdx.x;

    int offset = batch * off0 + head * off1 + seq1 * time2;
    int index = offset + seq2;
  

    if (seq2 < time2) {
        bool isTrue = mask[batch] / 4 >= index%time2 + 1? true: false;
        // 4 is subsampling number
        if(isTrue){
           dst[index] = ac[index] + bd[index];
        }
        else
        {
          dst[index] = T(-60000.0f);
        }
    }
}

template<typename T>
void invokeAddMat_Mask(T* output,
                  const float* matA, 
                const float* matB, 
                const int* mask,
                const int batch_size,
                const int seq_len,
                const int head_num,
                const int size_per_head,
                 cudaStream_t stream)
{ 
    dim3 grid_score(batch_size, head_num, seq_len);
    dim3 block_score(next_pow2(seq_len));
    addMat<<<grid_score, block_score, 0, stream>>>(output, matA, matB, mask, seq_len*seq_len*head_num, seq_len*seq_len, seq_len);
}
template void invokeAddMat_Mask(float* output, 
                        const float* matA, 
                        const float* matB, 
                            const int* mask,
                            const int batch_size,
                            const int seq_len,
                            const int head_num,
                            const int size_per_head,
                            cudaStream_t stream);