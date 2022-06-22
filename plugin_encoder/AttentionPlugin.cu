/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include "AttentionPlugin.h"

#include "NvInfer.h"

#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>

using namespace nvinfer1;

namespace
{
const char* Attention_PLUGIN_VERSION{"1"};
const char* Attention_PLUGIN_NAME{"Attention"};
} // namespace

// Static class fields initialization
PluginFieldCollection AttentionPluginCreator::mFC{};
std::vector<PluginField> AttentionPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(AttentionPluginCreator);

AttentionPlugin::AttentionPlugin(const std::string name)
    : mLayerName(name)
{

}

AttentionPlugin::AttentionPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{

}

const char* AttentionPlugin::getPluginType() const noexcept
{
    return Attention_PLUGIN_NAME;
}

const char* AttentionPlugin::getPluginVersion() const noexcept
{
    return Attention_PLUGIN_VERSION;
}

int AttentionPlugin::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs AttentionPlugin::getOutputDimensions(int32_t        outputIndex,
                                    const DimsExprs *   inputs,
                                    int32_t             nbInputs,
                                    IExprBuilder &      exprBuilder) noexcept
{
    
    assert(nbInputs == 14);
    assert(outputIndex == 0);

    return inputs[0];
}

int AttentionPlugin::initialize() noexcept
{

    return 0;
}

void AttentionPlugin::terminate() noexcept
{
    
}

size_t AttentionPlugin::getWorkspaceSize(const PluginTensorDesc *    inputDesc,
                            int32_t                     nbInputs,
                            const PluginTensorDesc *    outputs,
                            int32_t                     nbOutputs 
                            )   const noexcept
{

    int batch_size  = inputDesc[0].dims.d[0];    
    int seq_len    = inputDesc[0].dims.d[1];    
    int d_model     = inputDesc[0].dims.d[2]; 
    int buf_size    = batch_size * seq_len * d_model;
    int score_size = batch_size * seq_len * seq_len * d_model;
    int nElement = 12 * buf_size + 3 * score_size;
    
    size_t workspaceSize = nElement * sizeof(float);
    return workspaceSize;
}

void AttentionPlugin::attachToContext(cudnnContext * cudnn_handle,
                        cublasContext * cublas_handle,
                        IGpuAllocator * gpu_allocator
                        )noexcept
{
    cublasHandle_ = cublas_handle;
    gpu_allocator_ = gpu_allocator;
}

int AttentionPlugin::pre_enqueue(cudaStream_t stream) noexcept
{
   return 0;
}

int AttentionPlugin::enqueue(const PluginTensorDesc*  inputDesc,
                    const PluginTensorDesc* outputDesc,
                    const void *const *     inputs,
                    void *const *           outputs,
                    void *                  workspace,
                    cudaStream_t            stream) noexcept
{
    //cudaStreamSynchronize(stream);
    
    
    const int batch_size  = inputDesc[0].dims.d[0];    // B
    const int seq_len    = inputDesc[0].dims.d[1];    // T
    const int d_model     = inputDesc[0].dims.d[2];    // D
    const int head_num    = 4 ;
    const int size_per_head = 64 ;
    int i = 0;
    float *query_in                                = (float*)(inputs[i++]);
    int *mask                                      = (int*)(inputs[i++]);
    float *pos_emb                                 = (float*)(inputs[i++]);
    float *linear_pos_weight_kernel                = (float*)(inputs[i++]);
    float *query_weight_kernel   = (float*)(inputs[i++]);
    float *query_weight_bias     = (float*)(inputs[i++]);
    float *key_weight_kernel     = (float*)(inputs[i++]);
    float *key_weight_bias       = (float*)(inputs[i++]);
    float *value_weight_kernel   = (float*)(inputs[i++]);
    float *value_weight_bias     = (float*)(inputs[i++]);
    float *output_weight_kernel  = (float*)(inputs[i++]);
    float *output_weight_bias    = (float*)(inputs[i++]);
    float *pos_bias_u            = (float*)(inputs[i++]);
    float *pos_bias_v            = (float*)(inputs[i++]);
    int ws_offset  = 0;


    float* q_buf   = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len*d_model;
    float* q_bias  = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len*d_model;
    float* q_bias_t  = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len*d_model;
    float* q_bias_u  = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len*d_model;
    float* q_bias_v  = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len*d_model;
    float* p_buf   = (float*)workspace + ws_offset; ws_offset += 1*seq_len*d_model;
    float* p_buf_t   = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len*d_model;
    float* qk_buf  = (float*)workspace + ws_offset; ws_offset += batch_size*head_num*seq_len*seq_len;
    float* qk_buf_out  = (float*)workspace + ws_offset; ws_offset += batch_size*head_num*seq_len*seq_len;
    float* mat_ac_buf  = (float*)workspace + ws_offset; ws_offset += batch_size*head_num*seq_len*seq_len;
    float* mat_bd_buf  = (float*)workspace + ws_offset; ws_offset += batch_size*head_num*seq_len*seq_len;
    float* out_buf = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len*d_model;
    float* k_buf   = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len*d_model;
    float* k_bias  = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len*d_model;
    float* v_buf   = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len*d_model;
    float* v_bias  = (float*)workspace + ws_offset; ws_offset += batch_size*seq_len*d_model;

    
    cublasSetStream(cublasHandle_, stream);

    const float alpha = 1.0, beta = 0.0;
    cublasSgemm(cublasHandle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        d_model,                //  m
        batch_size*seq_len,    //  n
        d_model,                //  k
        &alpha, query_weight_kernel, d_model,
        (const float*)(query_in), d_model,
        &beta, q_buf, d_model
    );
    cublasSgemm(cublasHandle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        d_model,                //  m
        batch_size*seq_len,    //  n
        d_model,                //  k
        &alpha, key_weight_kernel, d_model,
        (const float*)(query_in), d_model,
        &beta, k_buf, d_model
    );
    cublasSgemm(cublasHandle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        d_model,                //  m
        batch_size*seq_len,    //  n
        d_model,                //  k
        &alpha, value_weight_kernel, d_model,
        (const float*)(query_in), d_model,
        &beta, v_buf, d_model
    );

    const int n = head_num * size_per_head;
    const int m = batch_size * seq_len;
    dim3 grid(m);
    dim3 block(min(n, 512));

    
    add_fusedQKV_bias_transpose_kernel<<<grid, block, 0, stream>>>(
        q_bias, q_buf, query_weight_bias, batch_size, seq_len, head_num, size_per_head);
    add_fusedQKV_bias_transpose_kernel<<<grid, block, 0, stream>>>(
        k_bias, k_buf, key_weight_bias, batch_size, seq_len, head_num, size_per_head);
    add_fusedQKV_bias_transpose_kernel<<<grid, block, 0, stream>>>(
        v_bias, v_buf, value_weight_bias, batch_size, seq_len, head_num, size_per_head);  // B, Head, Seqlen, size_per_head
    
    invokeTransposeQKV(q_bias_t,
                        q_bias,
                        batch_size,
                        seq_len,
                        head_num,
                        size_per_head,
                        stream
    );                                                                                     // B, Seqlen, H, size_per_head

    add_fusedQKV_bias_transpose_kernel<<<grid, block, 0, stream>>>(
        q_bias_u, q_bias_t, pos_bias_u, batch_size, seq_len, head_num, size_per_head);

    add_fusedQKV_bias_transpose_kernel<<<grid, block, 0, stream>>>(
        q_bias_v, q_bias_t, pos_bias_v, batch_size, seq_len, head_num, size_per_head);  // B, H, S, Size_per_head
 
    cublasSgemm(cublasHandle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        d_model,                //  m
        1*seq_len,    //  n
        d_model,                //  k
        &alpha, linear_pos_weight_kernel, d_model,
        (const float*)(pos_emb), d_model,
        &beta, p_buf, d_model
    );

    invokeTransposeQKV2(p_buf_t,
                        p_buf,
                        batch_size,
                        seq_len,
                        head_num,
                        size_per_head,
                        stream
    );                                                                // B, H, S, Size_per_head

    cudaStreamSynchronize(stream);
    
    cublasSgemmStridedBatched(
            cublasHandle_,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            seq_len,               //m,
            seq_len,               //n,
            size_per_head,          //k,
            &alpha,
            p_buf_t,                     //A,
            size_per_head,              //lda,
            seq_len*size_per_head,     //strideA,
            q_bias_v,                     //B,
            size_per_head,              //ldb,
            seq_len*size_per_head,     //strideB,
            &beta,
            mat_bd_buf,                 //C,
            seq_len,               //ldc,
            seq_len*seq_len,      //strideC,
            batch_size*head_num     //batchCount
        );
 
    cublasSgemmStridedBatched(
            cublasHandle_,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            seq_len,               //m,
            seq_len,               //n,
            size_per_head,          //k,
            &alpha,
            k_bias,                     //A,
            size_per_head,              //lda,
            seq_len*size_per_head,     //strideA,
            q_bias_u,                     //B,
            size_per_head,              //ldb,
            seq_len*size_per_head,     //strideB,
            &beta,
            mat_ac_buf,                 //C,
            seq_len,               //ldc,
            seq_len*seq_len,      //strideC,
            batch_size*head_num     //batchCount
        );
 
    invokeAddMat_Mask(
        qk_buf,
        mat_ac_buf,
        mat_bd_buf,
        mask,
        batch_size,
        seq_len,
        head_num,
        size_per_head,
        stream
    );
 
    float scalar = 1 / sqrtf(size_per_head * 1.0f);
   
    invokeSoftMax(qk_buf_out,
                        qk_buf,
                        batch_size,
                        seq_len,
                        seq_len,
                        head_num,
                        scalar,
                        stream
        );
    
    cublasSgemmStridedBatched(
            cublasHandle_,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            size_per_head,          //m,
            seq_len,               //n,
            seq_len,               //k,
            &alpha,
            v_bias,                     //A,
            size_per_head,              //lda,
            seq_len*size_per_head,     //strideA,
            qk_buf_out,                     //B,
            seq_len,                   //ldb,
            seq_len*seq_len,          //strideB,
            &beta,
            q_buf,                  //C,
            size_per_head,          //ldc,
            seq_len*size_per_head, //strideC,
            batch_size*head_num     //batchCount
        );
    
    invokeTransposeQKV(
        out_buf, q_buf, batch_size, seq_len, head_num, size_per_head, stream);
    
    cublasSgemm(cublasHandle_,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        d_model,                //  m
        batch_size*seq_len,    //  n
        d_model,                //  k
        &alpha, output_weight_kernel, d_model,
        (const float*)(out_buf), d_model,
        &beta, (float*)(outputs[0]), d_model
    );
    
    invokeAddBias((float*)(outputs[0]),     //  T* output, 
                        output_weight_bias,         //  const T* bias, 
                        batch_size * seq_len,       //  const int m, 
                        d_model,                    //  const int n, 
                        stream
                );
  
    int status = 0;
    return status;
}

size_t AttentionPlugin::getSerializationSize() const noexcept
{
    size_t ssize = 0;
    return ssize;
}

void AttentionPlugin::serialize(void* buffer) const noexcept
{

}

bool AttentionPlugin::supportsFormatCombination(int32_t               pos,
                                        const PluginTensorDesc *inOut,
                                        int32_t                 nbInputs,
                                        int32_t                 nbOutputs 
                                        ) noexcept
{
   
    bool res = false;

    if(inOut[pos].format != TensorFormat::kLINEAR)
        {
            return false;
        }

    
    switch(pos)
    {
    case 0:
        res = inOut[pos].type == DataType::kFLOAT && inOut[pos].dims.nbDims == 3; break;

    case 1:
        res = inOut[pos].type == DataType::kINT32 && inOut[pos].dims.nbDims == 1; break;

    case 2:
        res = inOut[pos].type == DataType::kFLOAT && inOut[pos].dims.nbDims == 3; break;
        
    case 3:
        res = inOut[pos].type == DataType::kFLOAT; break;

    case 4:
        res = inOut[pos].type == DataType::kFLOAT; break;

    case 5:        
        res = inOut[pos].type == DataType::kFLOAT; break; 
    case 6:
        res = inOut[pos].type == DataType::kFLOAT; break; 
    case 7:
        res = inOut[pos].type == DataType::kFLOAT; break; 
    case 8:
        res = inOut[pos].type == DataType::kFLOAT; break; 
    case 9:
        res = inOut[pos].type == DataType::kFLOAT; break; 
    case 10:
        res = inOut[pos].type == DataType::kFLOAT; break;
    case 11:
        res = inOut[pos].type == DataType::kFLOAT; break; 
    case 12:
        res = inOut[pos].type == DataType::kFLOAT; break; 
    case 13:
        res = inOut[pos].type == DataType::kFLOAT; break; 
    case 14:
        res = inOut[pos].type == DataType::kFLOAT; break;
    default:// should NOT be here
        res = false;
    }

    return res;
}

void AttentionPlugin::destroy() noexcept
{
    delete this;
}


IPluginV2DynamicExt* AttentionPlugin::clone() const noexcept
{
    
    auto plugin = new AttentionPlugin(mLayerName);
    plugin->setPluginNamespace(mNamespace.c_str());
    
    return plugin;
}

void AttentionPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    
    mNamespace = libNamespace;
}

const char* AttentionPlugin::getPluginNamespace() const noexcept
{
    
    return mNamespace.c_str();
}

AttentionPluginCreator::AttentionPluginCreator()
{
    
    mPluginAttributes.emplace_back(PluginField("AttentionType", nullptr, PluginFieldType::kCHAR, 4));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* AttentionPluginCreator::getPluginName() const noexcept
{
    
    return Attention_PLUGIN_NAME;
}

const char* AttentionPluginCreator::getPluginVersion() const noexcept
{
    
    return Attention_PLUGIN_VERSION;
}

const PluginFieldCollection* AttentionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* AttentionPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    return new AttentionPlugin(name);
}

IPluginV2DynamicExt* AttentionPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call AttentionPlugin::destroy()
    return new AttentionPlugin(name, serialData, serialLength);
}

void AttentionPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* AttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}