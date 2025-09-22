#!/usr/bin/env bash

conda create -n llama_hw python=3.11
conda activate llama_hw

cd ../data_preprocess
pip install -r requirements.txt
cd ../llama_training

# Modify this command depending on your system's environment.
# As written, this command assumes you have CUDA on your machine, but
# refer to https://pytorch.org/get-started/previous-versions/ for the correct
# command for your system.
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install tqdm==4.66.1
pip install requests==2.31.0
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install scikit-learn==1.2.2
pip install numpy==1.26.3
pip install tokenizers==0.13.3
pip install sentencepiece==0.1.99
pip install wandb