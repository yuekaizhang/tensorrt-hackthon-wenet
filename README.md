
# Build trt engine file
## Known Issue
* Offline ASR: For encoder, too large BxTxD size would encounter an illegal memory access  

## Env
We will deploy our model on Triton 22.05 therefore we here will use tensorrt 22.05 docker. Otherwise, you may would like using *LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tensorrt/lib* for triton.
```
docker run --gpus '"device=0"' -it --name trt_export -v <onnx model directory>:$onnx nvcr.io/nvidia/tensorrt:22.05-py3
```

## Quick Start
```
# Note: since onnx operations hard coding some ops name, you have to use this specfic wenet model and wenet repo.
# Otherwise, you may need to replaec the hard coding name by checking them through netron.   
# We would fix this in the future.

wget https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20211025_conformer_exp.tar.gz
tar zxvf 20211025_conformer_exp.tar.gz

git submodule update --init --recursive

python3 wenet/bin/export_onnx.py --config=$MODEL_DIR/train.yaml --checkpoint=$MODEL_DIR/final.pt --cmvn_file=$MODEL_DIR/global_cmvn --ctc_weight=0.1 --output_onnx_dir=$MODEL_DIR

bash build.sh --stage 0 --stop_stage 2 --trtexec_path <your_path/TensorRT-8.4.x.x/bin> --onnx_model_dir <onnx_model_parent_dir> --trt_plan_dir <export_trt_engine_dir> --workspace 20000

# --workspace specify the building phase available gpu memory in Mb, if you use tensorrt > 8.4, you don't need set this parameters.

```

## TODO 
* Modify export_onnx.py - [ ]
* streaming ASR model support - [ ]
* Add self-attention plugin fp16 support - [ ]
* int8 - [ ]
* Illegal memory access - [ ]