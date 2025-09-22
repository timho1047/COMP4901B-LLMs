"""Command-line interface helpers for llama training scripts."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser used by run_llama."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="train_100M",
        help="Directory containing raw text files for language model pretraining",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default="llama2-42M-babylm.pt",
        help="Path to a pretrained checkpoint for generation",
    )
    parser.add_argument("--max_sentence_len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--option",
        type=str,
        choices=("generate", "pretrain"),
        default="generate",
        help="Execution mode: 'generate' or 'pretrain'",
    )
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument(
        "--generated_sentence_low_temp_out",
        type=str,
        default="generated-sentence-temp-0.txt",
    )
    parser.add_argument(
        "--generated_sentence_high_temp_out",
        type=str,
        default="generated-sentence-temp-1.txt",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Optional wandb run display name")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity (team) name")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Effective batch size after gradient accumulation",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=None,
        help="Per-step batch size before gradient accumulation",
    )
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for pretraining")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of optimizer update steps to linearly warm up the learning rate",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Proportion of total update steps to warm up if warmup_steps is 0",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=512,
        help="Sequence length (in tokens) for pretraining batches",
    )
    parser.add_argument(
        "--tokenized_dir",
        type=str,
        default=None,
        help="Optional directory to cache tokenized pretraining data",
    )
    parser.add_argument(
        "--overwrite_tokenized",
        action="store_true",
        help="Force regeneration of cached tokenized data",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default=None,
        help="Optional validation data directory for pretraining",
    )
    parser.add_argument(
        "--val_tokenized_dir",
        type=str,
        default=None,
        help="Optional directory to cache tokenized validation data",
    )
    parser.add_argument(
        "--val_per_steps",
        type=int,
        default=0,
        help="Run validation every N training steps; 0 disables per-step validation",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="Optional test data directory for pretraining",
    )
    parser.add_argument(
        "--test_tokenized_dir",
        type=str,
        default=None,
        help="Optional directory to cache tokenized test data",
    )
    parser.add_argument(
        "--auto_resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Automatically resume training from <checkpoint>-resume.pt when available",
    )

    return parser


def parse_args(argv: Optional[Sequence[str]] = None):
    """Parse command-line arguments for run_llama."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.micro_batch_size is None:
        args.micro_batch_size = args.batch_size
    print(f"args: {vars(args)}")
    return args


__all__ = ["build_parser", "parse_args"]
