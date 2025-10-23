#!/usr/bin/env python3
"""VLLM-based inference script for IFEval benchmark.

This script loads a model using VLLM, applies the HuggingFace chat template,
generates responses for the IFEval prompts, and saves them in the required format.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_input_data(input_path: str) -> List[Dict]:
    """Load the IFEval input data from jsonl file.

    Args:
        input_path: Path to input_data.jsonl

    Returns:
        List of dictionaries with prompt and metadata
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(data)} prompts from {input_path}")
    return data


def apply_chat_template(
    prompts: List[str],
    tokenizer: AutoTokenizer,
    system_message: str = None
) -> List[str]:
    """Apply HuggingFace chat template to prompts.

    Args:
        prompts: List of user prompts
        tokenizer: HuggingFace tokenizer with chat template
        system_message: Optional system message to prepend

    Returns:
        List of formatted prompts ready for inference
    """
    formatted_prompts = []

    for prompt in prompts:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted)
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}. Using raw prompt.")
            formatted_prompts.append(prompt)

    return formatted_prompts


def run_inference(
    model_path: str,
    input_data_path: str,
    output_path: str,
    tensor_parallel_size: int = 1,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    batch_size: int = 32,
    system_message: str = None,
    gpu_memory_utilization: float = 0.9,
) -> None:
    """Run VLLM inference on IFEval dataset.

    Args:
        model_path: Path to the trained model
        input_data_path: Path to input_data.jsonl
        output_path: Path to save output responses in jsonl format
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy decoding)
        top_p: Nucleus sampling parameter
        batch_size: Batch size for inference
        system_message: Optional system message for chat template
        gpu_memory_utilization: GPU memory utilization fraction
    """
    logger.info("="*80)
    logger.info("Starting VLLM Inference for IFEval")
    logger.info("="*80)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Input data: {input_data_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Tensor parallel size: {tensor_parallel_size}")
    logger.info(f"Max tokens: {max_tokens}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Top-p: {top_p}")

    # Load input data
    input_data = load_input_data(input_data_path)
    prompts = [item["prompt"] for item in input_data]

    # Load tokenizer for chat template
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Apply chat template
    logger.info("Applying chat template to prompts")
    formatted_prompts = apply_chat_template(prompts, tokenizer, system_message)

    # Initialize VLLM
    logger.info("Initializing VLLM engine")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=None,  # Auto-detect from model config
    )

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Run inference
    logger.info("Running inference...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Save results
    logger.info(f"Saving results to {output_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, output in enumerate(outputs):
            result = {
                "prompt": prompts[i],  # Original prompt without chat template
                "response": output.outputs[0].text
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    logger.info("="*80)
    logger.info(f"Inference complete! Generated {len(outputs)} responses")
    logger.info(f"Results saved to: {output_path}")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="VLLM inference for IFEval benchmark"
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model"
    )
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Path to input_data.jsonl"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save output responses (jsonl format)"
    )

    # Optional arguments
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate per response"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy decoding)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (not used in VLLM, for compatibility)"
    )
    parser.add_argument(
        "--system_message",
        type=str,
        default=None,
        help="Optional system message to prepend to prompts"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction (0.0-1.0)"
    )

    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        input_data_path=args.input_data,
        output_path=args.output_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        system_message=args.system_message,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
