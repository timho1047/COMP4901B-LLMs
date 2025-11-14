RUN_NAME="q_3000

PATH_TO_THE_MERGED_MODLE="ckpt/$RUN_NAME/models/model_iter_0-merged"

bash scripts/self_train_gsm8k.sh Qwen3-0.6B \
    --run_name $RUN_NAME \
    --n_queries 3000

bash scripts/run_gsm8k_eval.sh $PATH_TO_THE_MERGED_MODLE \
    --output_dir results/${RUN_NAME}_1 \
    --temperature 0.6 \
    --max_tokens 512 \
    --top_p 0.95 \
    --top_k 20 \
    --n_rollouts 1 \

bash scripts/run_gsm8k_eval.sh \
    $PATH_TO_THE_MERGED_MODLE \
    --output_dir results/${RUN_NAME}_8 \
    --temperature 0.6 \
    --max_tokens 512 \
    --top_p 0.95 \
    --top_k 20 \
    --n_rollouts 8 \
    --n_queries 1000