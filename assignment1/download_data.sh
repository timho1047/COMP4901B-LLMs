#!/bin/bash

# Unified download script for COMP4901B Homework1
# Downloads both Common Crawl data and BabyLM datasets

echo "Starting data downloads..."

# Common Crawl data download (for data preprocessing task)
echo "Downloading Common Crawl data to data_preprocess directory..."
cd data_preprocess
ROOT=https://data.commoncrawl.org/crawl-data/CC-MAIN-2018-17/segments/1524125937193.1/

# Note: the --no-clobber arg required curl 7.83. If it doesn't work for you,
# you can either update curl or remove that argument.

curl -o data.warc.gz ${ROOT}warc/CC-MAIN-20180420081400-20180420101400-00000.warc.gz
gunzip data.warc.gz

curl -o data.wet.gz ${ROOT}wet/CC-MAIN-20180420081400-20180420101400-00000.warc.wet.gz
gunzip data.wet.gz

curl -o data.wat.gz ${ROOT}wat/CC-MAIN-20180420081400-20180420101400-00000.warc.wat.gz
gunzip data.wat.gz

echo "Common Crawl data download completed."
cd ..

# BabyLM dataset download (for LLaMA training task)
echo "Downloading BabyLM datasets to llama_training directory..."
cd llama_training

wget -O babylm_train.zip https://osf.io/download/ywea7/
unzip babylm_train.zip
wget -O babylm_test.zip https://osf.io/download/ftwu3/
unzip babylm_test.zip
wget -O babylm_dev.zip https://osf.io/download/uyd7v/
unzip babylm_dev.zip

echo "BabyLM dataset download completed."
cd ..

echo "All downloads finished successfully!"