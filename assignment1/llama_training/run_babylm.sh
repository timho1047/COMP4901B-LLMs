
# If you don't want to use WANDB for logging, simply ignore this line and the script will still print out metrics
export WANDB_API_KEY=YOUR_WANDB_API_KEY


python run_llama.py \
  --run_name run6-fix-loss \
  --option pretrain \
  --data_path train_100M \
  --block_size  256 \
  --batch_size 512 \
  --micro_batch_size 32 \
  --epochs 1 \
  --tokenized_dir train_100M/tokenized \
  --use_gpu  \
  --val_path dev \
  --val_tokenized_dir dev/tokenized \
  --val_per_steps 200 \
  --test_path  test \
  --test_tokenized_dir test/tokenized \
  --auto_resume \
  --warmup_ratio 0.1 \
  --lr 1e-3 
# --overwrite_tokenized # if you want to overwrite the tokenized data



