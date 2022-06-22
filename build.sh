#!/bin/bash

onnx_model_dir=/models/20211025_conformer_exp_onnx_export
trt_plan_dir=./
workspace=14000
stage=0
stop_stage=2

trtexec_path=/workspace/tensorrt/bin
. tools/parse_options.sh || exit 1;

export CUDA_VISIBLE_DEVICES=0

mkdir -p $trt_plan_dir

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then

     echo "repalce encoder ops with attention, layernorm plugin "
     python3 replaceAttention.py --input_onnx $onnx_model_dir/encoder.onnx --output_onnx ./encoder_attention.onnx || exit 1

     # echo "repalce decoder ops with attention, layernorm plugin "
     # python3 replaceAttention.py --input_onnx $onnx_model_dir/decoder.onnx --output_onnx ./decoder_attention.onnx || exit 1
     
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
     echo "compile attention, layernorm plugin"
     cd plugin_layernorm 
     make clean
     make
     cd -
     ln -s plugin_layernorm/LayerNormPlugin.so ./

     cd plugin_encoder
     make clean
     make
     cd -
     ln -s plugin_encoder/AttentionPlugin.so ./

     # cd plugin_decoder
     # make
     # make clean
     # cd -
     # ln -s plugin_decoder/DecoderAttentionPlugin.so ./
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
     echo "convert conformer encoder with attention layernorm plugin"
     # encoder seq_len 1024 would fail, decoder batch size 18 would be max
     $trtexec_path/trtexec \
          --onnx=./encoder_attention.onnx \
          --minShapes=speech:1x16x80,speech_lengths:1 \
          --optShapes=speech:64x512x80,speech_lengths:64 \
          --maxShapes=speech:64x512x80,speech_lengths:64 \
          --workspace=$workspace \
          --plugins=./LayerNormPlugin.so \
          --plugins=./AttentionPlugin.so \
          --saveEngine=$trt_plan_dir/encoder_attention.plan
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
     echo "convert decoder"
     $trtexec_path/trtexec   \
          --onnx=./decoder_attention.onnx \
          --minShapes=encoder_out:1x16x256,encoder_out_lens:1,hyps_pad_sos_eos:1x10x64,hyps_lens_sos:1x10,ctc_score:1x10 \
          --optShapes=encoder_out:4x64x256,encoder_out_lens:4,hyps_pad_sos_eos:4x10x64,hyps_lens_sos:4x10,ctc_score:4x10 \
          --maxShapes=encoder_out:16x256x256,encoder_out_lens:16,hyps_pad_sos_eos:16x10x64,hyps_lens_sos:16x10,ctc_score:16x10 \
          --workspace=$workspace \
          --plugins=./LayerNormPlugin.so \
          --plugins=./DecoderAttention.so \
          --saveEngine=$trt_plan_dir/decoder_attention.plan

fi