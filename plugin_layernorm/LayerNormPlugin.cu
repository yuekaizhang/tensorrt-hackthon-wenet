/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 #include "LayerNormPlugin.h"

using namespace nvinfer1;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

using kv_float = cub::KeyValuePair<float, float>;
using kv_half = cub::KeyValuePair<half, half>;
__device__ inline kv_float operator+(const kv_float& a, const kv_float& b)
{
    return kv_float(a.key + b.key, a.value + b.value);
}
__device__ inline half2 __hadd2_with_fallback(const half2 a, const half2 b)
{
#if __CUDA_ARCH__ >= 530
    return __hadd2(a, b);
#else
    float2 out{};
    out.x = __half2float(a.x) + __half2float(b.x);
    out.y = __half2float(a.y) + __half2float(b.y);
    return __float22half2_rn(out);
#endif
}
__device__ inline kv_half operator+(const kv_half& a, const kv_half& b)
{
    const half2 a2 = __halves2half2(a.key, a.value);
    const half2 b2 = __halves2half2(b.key, b.value);
    const half2 res = __hadd2_with_fallback(a2, b2);
    return kv_half(res.x, res.y);
}
template <typename T>
using kvp = cub::KeyValuePair<T, T>;

template <typename T, typename R, typename P, int TPB>
__device__ inline void layerNorm(
    const kvp<R>& threadData, const int ld, const int offset, T* output)
{
    // Assuming threadData is already divided by ld

    using BlockReduce = cub::BlockReduce<kvp<R>, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ R mu;     // mean
    __shared__ R rsigma; // 1 / std.dev.

    const auto sumKV = BlockReduce(temp_storage).Reduce(threadData, cub::Sum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.key;
        rsigma = rsqrt(sumKV.value - mu * mu);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        const int idx = offset + i;
        const R val = output[idx];
        output[idx] = (val - mu) * rsigma;
    }
}


template <typename T, unsigned TPB, bool hasBias>
__global__ void LayerNormKernel(
    const int ld, const T* input, T* output, const float bias)
{
    const T rld = T(1) / T(ld);
    const int offset = blockIdx.x * ld;

    cub::Sum pairSum;
    // reduce x and x^2
    kvp<T> threadData(0, 0);

    for (int i = threadIdx.x; i < ld; i += TPB)
    {
        const int idx = offset + i;
        T val = T(input[idx]);

        if (hasBias)
        {
            val += T(bias);
        }
        const T rldval = rld * val;
        threadData = pairSum(threadData, kvp<T>(rldval, rldval * val));
        output[idx] = val;
    }

    layerNorm<T, T, T, TPB>(threadData, ld, offset, output);
}

template <typename T, bool hasBias>
int computeLayerNorm(cudaStream_t stream, const int ld, const int n, const T* input, T* output, const float bias)
{

    // this must be true because n is the total size of the tensor
    assert(n % ld == 0);
    const int gridSize = n / ld;
    switch (ld)
        {
        case 256: 
            //constexpr int blockSize = 256;
            LayerNormKernel<T, 256, hasBias>
        <<<gridSize, 256, 0, stream>>>(ld, input, output, bias);
            break;
        case 512:
            LayerNormKernel<T, 512, hasBias>
        <<<gridSize, 512, 0, stream>>>(ld, input, output, bias);
            break;
        default:
            printf("[LayerNormPlugin::enqueue] hidden dim = %d is not supported\n",ld);
            break;
        }

    
    return 0;
}


int LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int inputVolume = volume(inputDesc[0].dims);
    size_t mLd = inputDesc[0].dims.d[2]; // assert [Batch,Len,Hidden] 
    int status = -1;
    DataType iType = inputDesc->type;
    // Our plugin outputs only one tensor
    // Launch CUDA kernel wrapper and save its return value
    if (iType == DataType::kFLOAT)
    {
        const auto* const input = static_cast<const float*>(inputs[0]);
      
        auto* output = static_cast<float*>(outputs[0]);

        status = computeLayerNorm<float, false>(stream, static_cast<int>(mLd), inputVolume, input, output, bias_);
        
    }
    else if (iType == DataType::kHALF)
    {
        const auto* const input = static_cast<const half*>(inputs[0]);
      
        auto* output = static_cast<half*>(outputs[0]);

        status = computeLayerNorm<half, false>(stream, static_cast<int>(mLd), inputVolume, input, output, bias_);
      
    }
    else
    {
        std::cout << "Unsupported type error, expected [kHALF,kFLOAT], but received " << static_cast<int>(iType) << "." << std::endl;
        assert(false);
    }
    return status;
}


REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);
