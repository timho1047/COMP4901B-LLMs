export CUDA_VISIBLE_DEVICES=5


bash scripts/train_eval.sh

RUN_NAME="default_multi"

bash scripts/self_train_gsm8k.sh Qwen3-0.6B \
    --run_name $RUN_NAME \
    --num_iterations 4

for i in {1..4}; do
    PATH_TO_THE_MERGED_MODLE="ckpt/$RUN_NAME/models/model_iter_${i}-merged"

    bash scripts/run_gsm8k_eval.sh $PATH_TO_THE_MERGED_MODLE \
    --output_dir results/${RUN_NAME}_iter_${i}_1 \
    --temperature 0.6 \
    --max_tokens 512 \
    --top_p 0.95 \
    --top_k 20 \
    --n_rollouts 1

    bash scripts/run_gsm8k_eval.sh $PATH_TO_THE_MERGED_MODLE \
        --output_dir results/${RUN_NAME}_iter_${i}_8 \
        --temperature 0.6 \
        --max_tokens 512 \
        --top_p 0.95 \
        --top_k 20 \
        --n_rollouts 8 \
        --n_queries 1000
done