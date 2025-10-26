#!/bin/bash

export VLLM_CACHE_ROOT=.vllm_cache

MODEL_PATH=$1
OUTPUTPATH=$2

python run_ifeval.py \
    --mode all \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUTPATH \
    --max_tokens 1024 \