export CUDA_VISIBLE_DEVICES=5

RUN_NAME="bsz_256"

PATH_TO_THE_MERGED_MODLE="ckpt/$RUN_NAME/models/model_iter_1-merged"

bash scripts/self_train_gsm8k.sh Qwen3-0.6B \
    --run_name $RUN_NAME \
    --total_batch_size 256

bash scripts/run_gsm8k_eval.sh $PATH_TO_THE_MERGED_MODLE \
    --output_dir results/${RUN_NAME}_1 \
    --temperature 0.6 \
    --max_tokens 512 \
    --top_p 0.95 \
    --top_k 20 \
    --n_rollouts 1

bash scripts/run_gsm8k_eval.sh $PATH_TO_THE_MERGED_MODLE \
    --output_dir results/${RUN_NAME}_8 \
    --temperature 0.6 \
    --max_tokens 512 \
    --top_p 0.95 \
    --top_k 20 \
    --n_rollouts 8 \
    --n_queries 1000

########################################################

RUN_NAME="bsz_64"

PATH_TO_THE_MERGED_MODLE="ckpt/$RUN_NAME/models/model_iter_1-merged"

bash scripts/self_train_gsm8k.sh Qwen3-0.6B \
    --run_name $RUN_NAME \
    --total_batch_size 64

bash scripts/run_gsm8k_eval.sh $PATH_TO_THE_MERGED_MODLE \
    --output_dir results/${RUN_NAME}_1 \
    --temperature 0.6 \
    --max_tokens 512 \
    --top_p 0.95 \
    --top_k 20 \
    --n_rollouts 1

bash scripts/run_gsm8k_eval.sh $PATH_TO_THE_MERGED_MODLE \
    --output_dir results/${RUN_NAME}_8 \
    --temperature 0.6 \
    --max_tokens 512 \
    --top_p 0.95 \
    --top_k 20 \
    --n_rollouts 8 \
    --n_queries 1000