from huggingface_hub import snapshot_download

snapshot_download(repo_id='Qwen/Qwen3-0.6B', repo_type='model', local_dir='Qwen3-0.6B', cache_dir='.cache')
snapshot_download(repo_id='gsm8k', repo_type='dataset', local_dir='gsm8k-local', cache_dir='.cache')
# snapshot_download(repo_id='unsloth/mistral-7b-bnb-4bit', repo_type='model', local_dir='mistral-7b-bnb-4bit', cache_dir='.cache')