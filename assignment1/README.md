# COMP4901B Homework 1 — Baby LLaMA2 Pretraining + Data Pipeline

**<span style="color: red;">Due Date: 2025-10-08 23:59:59</span>**

**Full score: 100 points.**

This repository contains the starter code and tooling for Homework 1. You will:
- Build a small-scale data preprocessing pipeline over a Common Crawl WARC shard dataset and a toy topic dataset for pretraining language models.
- Implement core LLaMA2 components and pretrain a 42M‑parameter model on the BabyLM corpora.
- Implement and run text generation (temperature sampling) from the language model and analyze outputs.

You can either use your own laptop/machine if they have GPUs or use free Google Colab ([Getting Started With Google Colab: A Beginners Guide](https://www.marqo.ai/blog/getting-started-with-google-colab-a-beginners-guide).) to perform the experiments.

This writeup is organized as follows:
- [Rules](#rules) is the base rule
- [Your Tasks](#your-tasks) is the main part and contain all the tasks you should complete for this homework
- [Setup](#setup) is the basic setup for this codebase
- [Running the Pipelines](#running-the-pipelines) is the detailed instructions for using the code
- [Important Notes](#important-notes) are some notes about using the code
- [Submission](#submission) contains instructions on how to submit your homework
- [Quick Reference](#quick-reference)
- [Repo Structure](#repo-structure)
- [Troubleshooting](#troubleshooting) are some helpful tips. You are welcome to ask questions on Canvas as well.

## Rules
Detailed course logistics is at [here](https://docs.google.com/document/d/1mWm_TYYQpD3NpJISlFQGurBIXxWjizEc1zVffIDhiXU/edit?usp=sharing). 

1. You need to work on the homework independently, without collaborating with other humans.
2. You are allowed to collaborate with other AIs. Actually you are encouraged to use AI tools, because this codebase may involve some parameters or tools that you are not familiar with -- and we certainly cannot teach everything during the lectures. Generally AI tools can help a lot explaining them.
3. As noted in the logistics, each student will have a total of three free late (calendar) days to use for homeworks. Once these late days are exhausted, any assignments turned in late will be penalized 20% per late day. However, no assignment will be accepted more than three days after its due date.
4. For questions about his homework, please post on the Canvas 

Honor code violation will be directly cause failing the course.


## Your Tasks

### 0. (5 Points) Data download 
- Run `./download_data.sh` from the repo root to fetch both Common Crawl files (WARC/WET/WAT) and the BabyLM datasets.
- The script writes CC files into `data_preprocess/` and BabyLM into `llama_training/`.

### 1. (15 points) Part 1 — Data preprocessing (data_preprocess/) 
(10 pts) 2.1. Implement the TODOs in `data_preprocess/homework.py` (you may refer to the dataset in 2.2 below to understand what the data to be preprocessed looks like):
  - `html_to_text`, `replace_pii`, `clean_text`, `heuristic_quality_filter`, `is_english_text`, `deduplicate_texts`.
  
(5 pts) 2.2. From `data_preprocess/`, run the pipeline to filter the WARC dump and deduplicate the topic dataset:
  - `cd data_preprocess`
  - `python homework.py --fname data.warc --output cleaned_test.txt --dfname topic_dataset.json`
- Report (paste into the PDF):
  - Number of WARC records processed and the number that pass all filters.
  - Number of topic dataset items after deduplication.

### 2. (40 points) Part 2 — LLaMA2 pretraining (llama_training/)
(20pts) 2.1 Fill the following TODO items:
  - The TODO in `Attention` class in `llama_training/llama.py`,
  - TODO in `llama_training/run_llama.py` (warmup LR scheduler),

**Supplementary for warmup LR scheduler implementation:**

Learning rate warmup is a training technique that gradually increases the learning rate from 0 to the target learning rate over a specified number of steps. This helps stabilize training in the early stages, especially for large models and batch sizes.

The warmup scheduler you need to implement follows this pattern:

```
Learning Rate
      │
      │     ┌─────────────────────────────── Target LR (constant phase)
      │    ╱
      │   ╱
      │  ╱  (linear warmup phase)
      │ ╱
      │╱
      └─────────────────────────────────────► Training Steps
      0    warmup_steps                      total_steps
```

**Implementation Details:**
- **Warmup Phase** (steps 0 to `warmup_steps`): Learning rate increases linearly from 0 to `base_lr`
- **Constant Phase** (steps > `warmup_steps`): Learning rate remains at `base_lr`

**Why Use Warmup?**
1. **Prevents early instability**: Large learning rates at the start can cause gradient explosions
2. **Better convergence**: Gradual increase allows the model to find a good initial direction
3. **Common in transformer training**: Essential for training large language models effectively


(20pts) 2.2 Run the BabyLM pretraining pipeline. A reference command is provided in `llama_training/run_babylm.sh`; feel free to edit it to fit your environment. You can refer [here](#pretraining-part-2) for the meaning of the arguments in the script.
- Train for at least 200–1000 optimizer steps. On Colab, expect roughly 40–200 minutes depending on settings.
- Report (paste into the PDF):
  - Training setup (GPU/CPU, effective batch size, sequence length, LR, warmup, steps).
  - Loss curve (per‑token loss vs. updates). A screenshot from your console or wandb is fine.

**Note on Logging and Visualization**: The training script will automatically attempt to use [Weights & Biases (wandb)](https://wandb.ai) to log and plot training statistics including losses, learning rates, and validation metrics. This provides beautiful real-time dashboards and automatic plot generation. However, **wandb is completely optional** - if you don't set up a wandb account or API key, the training script will gracefully skip wandb logging and continue working normally. All metrics will still be printed to the console. If you choose not to use wandb, you may need to manually create loss curve plots from the console output for your report. If you want to try using wandb, you can check [wandb quickstart](https://docs.wandb.ai/quickstart) and [Important Notes](#important-notes) for more details.

### 3. (40 points) Part 3 — Generation & analysis
(20pts) 3.1 Implement `generate()` in `llama_training/llama.py` (greedy for `temp=0.0`, temperature sampling and top‑k for `temp>0`).

**Supplementary for problem 3.1: Background on different sampling methods**

As we learned in class, at inference time, for each step we predict the next token. This is achieved by sampling from a probability distribution over all tokens in the vocabulary, and this probability distribution is computed from a softmax function:

$$P(\text{next token} = w_i) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

where:
- $z_i$ is called the logit for token $i$
- $V$ is the vocabulary size

In practice, however, we can add a temperature parameter $T$ to the softmax function to control the randomness of the sampling:

$$P(\text{next token} = w_i) = \frac{e^{z_i/T}}{\sum_{j=1}^{V} e^{z_j/T}}$$

When $T<1$, then the distribution becomes sharper, and sampling is more deterministic and less diverse. When $T>1$, then the distribution becomes more uniform, and sampling is more random and more diverse. You can think that when $T\rightarrow0$, then the distribution becomes a one-hot distribution and the model will always produce the most likely token, and when $T\rightarrow\infty$, then the distribution becomes a uniform distribution.

Different sampling strategies in language models include:
- **Greedy sampling** ($T \to 0$): Always pick the highest probability token
- **Temperature sampling** ($T > 0$): Sample from the softmax distribution, where higher $T$ increases randomness
- **Top-k sampling**: Restrict sampling to the $k$ most likely tokens before applying softmax. This is like masking the tokens outside the top $k$ most likely tokens. In class, we have learned how to do attention masking, but here the implementation should be quite similar since they are both softmax. Top-K sampling is also called nucleus sampling.



(20pts) 3.2 Generate using two temperatures `temp=0.0` and `temp=0.5` with temperate sampling and compare:
  - `cd llama_training`
  - `python run_llama.py --pretrained-model-path YOUR_TRAINED_MODEL.pt --option generate`
  - By default this writes `generated-sentence-temp-0.txt` and `generated-sentence-temp-1.txt`.
  - Also try generating with our provided fully trained model:
  - https://huggingface.co/yuzhen17/llama2-42M-babylm (download the checkpoint locally and pass its path to `--pretrained-model-path`).
- Report (paste into the PDF):
  - (10pts) Generated outputs for `temp=0.0` and `temp=0.5` from both your trained model and our provided model.
  - (10pts) Explain which temperate is better and why (e.g., diversity vs. coherence).


## Setup

(If you are using the colab, you might skip this part) 

Install the running env using 
  - `cd llama_training`
  - `bash setup.sh`
  - Adjust the CUDA line in `setup.sh` to match your hardware (see https://pytorch.org/get-started/previous-versions/).



## Running the Pipelines

### Data preprocessing (Part 1)
- Work from `data_preprocess/` so relative paths resolve (e.g., `bad_word_list.txt`).
- Typical command:
  - `python homework.py --fname data.warc --output cleaned_test.txt --dfname topic_dataset.json`
- Tips:
  - Use `--num_records` to sample a small subset while debugging.
  - Keep new artifacts in `data_preprocess/` (e.g., `cleaned_*.txt`).

### Pretraining (Part 2)
- Recommended: run this part on a GPU machine (e.g., [Google Colab](https://colab.research.google.com/)). A minimal demo notebook is available at `llama_training/colab_demo.ipynb`.
- Reference script: `llama_training/run_babylm.sh`
- Key arguments (all provided to `python llama_training/run_llama.py`):

#### **Data and Tokenization Arguments**
  - `--data_path` (default: `"train_100M"`): Directory containing raw text files for language model pretraining. The system looks for files with extensions `*.train`, `*.val`, `*.test`, `*.txt`, `*.dev` in this directory. Each text file is processed line by line, with each line being tokenized separately and concatenated into a single token stream.
  
  - `--tokenized_dir` (default: `None`, falls back to `<data_path>/tokenized`): Directory to cache pre-tokenized data as binary files. Raw text files are converted to compact `uint16` token arrays stored as `.bin` files. A `metadata.json` file tracks which source files correspond to which token files and their token counts. This dramatically speeds up subsequent training runs since tokenization only needs to happen once.
  
  - `--overwrite_tokenized` (flag): Forces regeneration of cached tokenized data. When enabled, ignores existing tokenized cache and re-tokenizes all text files. Useful when you change tokenizer settings or update the source text files.

#### **Sequence and Batch Configuration**
  - `--block_size` (default: `512`): Sequence length in tokens for each training example. Each training sequence contains exactly this many tokens. The dataset splits longer tokenized text into non-overlapping chunks of this size. Shorter sequences save GPU memory but may reduce model quality. The code uses autoregressive language modeling where tokens 0 to N-1 predict tokens 1 to N.
  
  - `--batch_size` (default: `8`): Effective batch size after gradient accumulation. This is the total number of sequences processed before applying one optimizer update. The actual implementation uses gradient accumulation to achieve this target even if GPU memory can't fit the full batch.
  
  - `--micro_batch_size` (default: same as `batch_size`): Per-step batch size before gradient accumulation. Number of sequences processed in each forward/backward pass. If `micro_batch_size < batch_size`, gradients are accumulated across multiple micro-batches before updating parameters. For example, if `batch_size=512` and `micro_batch_size=32`, gradients accumulate over 16 steps.

  **Gradient Accumulation (this is optional for this course. You can just set `micro_batch_size` the same as `batch_size` if you don't want to use gradient accumulation)**: This technique allows training with large effective batch sizes even when GPU memory is limited. Instead of processing all sequences at once, the model processes smaller micro-batches sequentially, accumulating (summing) the gradients from each micro-batch without updating the model parameters. Only after processing enough micro-batches to reach the target `batch_size` does the optimizer actually update the model weights using the accumulated gradients. This is mathematically equivalent to training with the full batch size but uses less GPU memory. The trade-off is slightly increased training time due to the sequential processing, but it enables training larger models or using larger effective batch sizes on resource-constrained hardware.

#### **Training Schedule**
  - `--epochs` (default: `5`): Number of complete passes through the tokenized training corpus. After each epoch, the model saves a checkpoint. Training progress is tracked by both epoch number and global step count (total optimizer updates across all epochs).
  
  - `--lr` (default: `1e-3`): Base learning rate for the optimizer. Uses AdamW optimizer. The actual learning rate follows the warmup schedule implemented in `WarmupLearningRateScheduler`.
  
  - `--warmup_steps` (default: `0`): Number of optimizer steps for linear learning rate warmup. Learning rate starts at 0 and linearly increases to `--lr` over this many steps, then stays constant. Takes precedence over `--warmup_ratio` if both are specified.
  
  - `--warmup_ratio` (default: `0.0`): Alternative to `--warmup_steps` - fraction of total training steps to use for warmup. If `--warmup_steps=0`, this calculates warmup steps as `total_expected_updates * warmup_ratio`. For example, with 1000 total steps and `warmup_ratio=0.1`, warmup lasts 100 steps.

#### **Validation and Testing**
  - `--val_path` (default: `None`): Directory containing validation text files. Similar structure to `--data_path`. When provided, validation runs during training to monitor model performance and save the best checkpoint.
  
  - `--val_tokenized_dir` (default: `None`, falls back to `<val_path>/tokenized`): Cache directory for tokenized validation data. Same caching mechanism as training data. Keeps validation tokenization separate from training.
  
  - `--val_per_steps` (default: `0`): Run validation every N optimizer update steps. When > 0, triggers validation evaluation after every N global steps. Validation loss and perplexity are logged, and the best checkpoint (lowest validation loss) is automatically saved. Setting to 0 disables periodic validation.
  
  - `--test_path` (default: `None`): Directory containing test text files for optional final evaluation.
  
  - `--test_tokenized_dir` (default: `None`): Cache directory for tokenized test data. Test evaluation runs only once at the very end of training, after all epochs complete. Results are logged but don't affect checkpointing.

#### **Checkpointing and Resume**
  - `--auto_resume` (default: `False`): Automatically resume from interrupted training. Creates two types of checkpoints: (1) Regular checkpoints named like `<run_name>-pretrain-<epochs>-<lr>.pt`, saved after each epoch and when validation improves; (2) Rolling resume checkpoints named `*-resume.pt`, updated throughout training for robust restart capability. When enabled, training automatically detects and loads from the resume checkpoint if it exists.
  
  - `--run_name` (default: `None`): Optional name that affects the default checkpoint filename. If not provided, defaults to "run". The final checkpoint path becomes `<run_name>-pretrain-<epochs>-<lr>.pt`.

#### **System Configuration**
  - `--use_gpu` (flag): Run training on CUDA if available. Without this flag, training runs on CPU, which is much slower but doesn't require GPU resources.
#### **Additional Notes**
- **Tokenization caching**: First run converts text files into compact `uint16` token bins under `tokenized/` and writes `metadata.json`. Reuse the cache across runs to avoid re-tokenizing.
- **Checkpointing**: Checkpoints are named like `<run_name>-pretrain-<epochs>-<lr>.pt` by default. Best‑on‑val is saved when validation loss decreases. With `--auto_resume`, a rolling `*-resume.pt` is updated for robust restarts.

### Generation (Part 3)
- The `generate` mode loads a `.pt` checkpoint and produces two files by default:
  - `generated-sentence-temp-0.txt` (greedy, `temp=0.0`)
  - `generated-sentence-temp-1.txt` (stochastic, `temp=0.5`)
- You can tweak the hardcoded `prefix` in `llama_training/run_llama.py` if desired.


## Important Notes

- Environment
  - Use `bash llama_training/setup.sh` to create the `llama_hw` env and install PyTorch. Edit the CUDA channel/version to match your machine.
  - If you don’t have a GPU, you can omit `--use_gpu`, but training will be very slow.
- Memory saving
  - Reduce `--block_size` and/or `--micro_batch_size`; gradient accumulation will preserve the effective `--batch_size`.
  - Close notebooks/tabs and lower `--n_layers` or `dim` only if you modify the model, but this is not required for the assignment.
- Validation and save frequency
  - `--val_per_steps N` runs validation every N optimizer updates. The script saves the best checkpoint when validation improves and also at epoch boundaries.
- Pretraining initialization
  - If the file passed as `--pretrained-model-path` exists (default `llama2-42M-babylm.pt` from setup), training will initialize from it; otherwise it trains from scratch. Change or remove the path to control this behavior.
- wandb logging
  - Set `export WANDB_API_KEY=...` and pass `--wandb_project`/`--wandb_entity` to enable logging (see `llama_training/run_babylm.sh`).
  - To disable wandb, don’t set a project/entity; training will still print metrics.
- Data sizes
  - The Common Crawl files are large. While iterating, use `--num_records` to test quickly.
- Paths and outputs
  - Keep generated artifacts in their producing directory (e.g., cleaned text in `data_preprocess/`, token bins under `<split>/tokenized`).


## Submission

Submit a zip of the codebase (only the `assignment1` directory) and a PDF report to Canvas. Your PDF should include your full name, student ID, and your UST email. For the contents:
- Part 1: WARC filtering and topic deduplication
  - Command used.
  - Number of records processed and passed; number after deduplication.
  - Description of your cleaning/heuristic/English detection logic.
- Part 2: Pretraining
  - Commands (either `run_babylm.sh` or the equivalent `python` command line).
  - Training configuration (GPU/CPU, block size, batch size, micro batch size, LR, warmup, total steps/epochs).
  - Training loss curve (per‑token loss vs. steps) and any validation curves.
  - Notes on stability and any changes you made to save memory or resume.
- Part 3: Generation
  - Command used and the checkpoint path.
  - Generated outputs (`temp=0.0` and `temp=0.5`) from both your trained model and our provided model .
  - Your explanation of which is better and why (coherence vs. diversity; effect of temperature and top‑k).

Also include:
- Any edits you made to `run_babylm.sh` (parameters and rationale).

## Quick Reference

- Download data: `./download_data.sh`
- Preprocess (from `data_preprocess/`): `python homework.py --fname data.warc --output cleaned_test.txt --dfname topic_dataset.json`
- Pretrain: `bash llama_training/run_babylm.sh` (edit parameters as needed)
- Generate: `python llama_training/run_llama.py --pretrained-model-path <ckpt>.pt --option generate`


## Repo Structure

- `data_preprocess/`: WARC/topic cleaners and helpers. Keep outputs here.
- `llama_training/`: LLaMA model, optimizer, training scripts, tokenizer, configs, and BabyLM data.
- Root: zipped raw archives (if any), this README, and `download_data.sh`.


## Troubleshooting

- If you want to understand the actual variables and inputs to the function during runtime (e.g., what is the size of the tensor) to help your implementation, the best practice is to use [pdb](https://docs.python.org/3/library/pdb.html) to set breakpoints or simply print out the variables.
- “No pretraining sequences available” — Ensure your `block_size` is not larger than the number of tokens per file and that `tokenized/metadata.json` exists. Delete or pass `--overwrite_tokenized` to rebuild if needed.
- CUDA OOM — Lower `--micro_batch_size` and/or `--block_size`.
- Slow tokenization — Use `--tokenized_dir` and reuse caches; only retokenize when text changes.
- Checkpoint not loading — Verify the path to `.pt` is correct and was saved by this codebase. For HF, download the actual `.pt` file locally.
- More about Google Colab: [Getting Started With Google Colab: A Beginners Guide](https://www.marqo.ai/blog/getting-started-with-google-colab-a-beginners-guide).
