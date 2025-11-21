"""
Evaluation script for AIME predictions using math-verify with pass@k and avg@k metrics
"""

import argparse
import json
import itertools
from typing import Any, List, Dict, Union
from collections import defaultdict
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
import sympy
import numpy as np


def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.

    Args:
        num_samples: Number of samples generated per problem
        num_correct: Number of correct samples per problem
        k: The k value for pass@k metric

    Returns:
        Array of pass@k estimates for each problem
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])




def group_by_problem(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group results by problem ID.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary mapping problem ID to list of rollouts
    """
    grouped = defaultdict(list)
    for result in results:
        grouped[result['id']].append(result)
    return grouped


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate AIME answers using sympy and math-verify')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input JSONL file containing model outputs')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output JSONL file for evaluation results')
    parser.add_argument('--gold_is_latex', action='store_true',
                       help='Use basic latex normalization for gold answers (default: False)')
    return parser.parse_args()


def load_jsonl_data(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load and validate JSONL data.

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        List of dictionaries containing problem data
    """
    try:
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    entry = json.loads(line)
                    # Validate required fields
                    if 'llm_response' not in entry or 'answer' not in entry:
                        raise ValueError(
                            f"Line {line_num}: JSONL entries must contain 'llm_response' and 'answer' fields"
                        )
                    data.append(entry)
        return data
    except Exception as e:
        raise Exception(f"Error loading JSONL file: {str(e)}")


def save_jsonl_data(data: List[Dict[str, Any]], jsonl_path: str):
    """Save data to JSONL file.

    Args:
        data: List of dictionaries to save
        jsonl_path: Path to output JSONL file
    """
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def serialize_sympy_object(obj: Any) -> str:
    """Convert sympy object to string representation.

    Args:
        obj: Sympy object or other value

    Returns:
        String representation
    """
    if obj is None:
        return ""
    try:
        if isinstance(obj, (list, tuple)):
            # Deduplicate while preserving order
            seen = set()
            unique_items = []
            for x in obj:
                str_x = str(x) if x is not None else ""
                if str_x and str_x not in seen:
                    seen.add(str_x)
                    unique_items.append(str_x)
            return ", ".join(unique_items) if unique_items else ""
        return str(obj)
    except Exception as e:
        return f"Error: {str(e)}"


def process_answers(data: List[Dict[str, Any]], gold_is_latex: bool) -> tuple:
    """Process each answer through the sympy extraction workflow and compare with gold using math_verify.

    Args:
        data: List of dictionaries with 'llm_response' and 'answer' fields
        gold_is_latex: Whether gold answers are in LaTeX format

    Returns:
        List of result dictionaries with evaluation results
    """
    results = []
    correct_count = 0
    total_count = 0

    # Create the verification function
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig() if gold_is_latex else ExprExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        aggregation_function=max,
        precision=6
    )

    for entry in data:
        extracted_answers = None
        gold_answers = None
        grade = 0
        error = None

        try:
            # Use the verification function
            # Note: verify_func expects (gold_list, pred_list)
            grade, extracted_answers = verify_func([entry['answer']], [entry['llm_response']])

            if extracted_answers is None:
                extracted_answers = None
                gold_answers = None
            else:
                # extracted_answers is a tuple of (gold_extracted, pred_extracted)
                gold_answers = extracted_answers[0]
                extracted_answers = extracted_answers[1]

            total_count += 1
            if grade == 1:
                correct_count += 1

        except Exception as e:
            error = str(e)
            total_count += 1

        # Build result entry (preserve all original fields + add evaluation)
        result = {**entry}  # Copy all original fields
        result.update({
            'extracted_answer': serialize_sympy_object(extracted_answers),
            'extracted_gold': serialize_sympy_object(gold_answers),
            'is_correct': grade == 1,
        })

        if error:
            result['error'] = error

        results.append(result)

    # Calculate basic metrics
    accuracy = correct_count / total_count if total_count > 0 else 0

    # Group results by problem ID for pass@k and avg@k calculations
    grouped = group_by_problem(results)
    num_problems = len(grouped)

    # Initialize metrics
    metrics = {
        'total_count': total_count,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'num_problems': num_problems,
        'avg_rollouts_per_problem': total_count / num_problems if num_problems > 0 else 0
    }

    # Calculate pass@k and avg@k if there are multiple rollouts
    if num_problems < total_count:  # Multiple rollouts detected
        # Calculate per-problem statistics for pass@k
        num_samples_list = []
        num_correct_list = []

        for _, problem_results in grouped.items():
            num_samples = len(problem_results)
            num_correct = sum(1 for r in problem_results if r.get('is_correct', False))
            num_samples_list.append(num_samples)
            num_correct_list.append(num_correct)

        num_samples_array = np.array(num_samples_list)
        num_correct_array = np.array(num_correct_list)

        # Automatically determine k values based on minimum rollouts per problem
        min_rollouts = int(np.min(num_samples_array))
        k_values = list(range(1, min_rollouts + 1))

        # Calculate pass@k for each k value
        passk_metrics = {}

        for k in k_values:
            # pass@k: only for problems with >= k samples
            valid_indices = num_samples_array >= k

            if np.any(valid_indices):
                passk_scores = estimate_pass_at_k(
                    num_samples_array[valid_indices],
                    num_correct_array[valid_indices],
                    k
                )
                passk_metrics[f'pass@{k}'] = float(np.mean(passk_scores))
            else:
                passk_metrics[f'pass@{k}'] = None

        metrics['pass_at_k'] = passk_metrics
        metrics['k_values'] = k_values

    return results, metrics


def main():
    args = parse_args()

    # Load input JSONL
    print(f"Loading data from {args.input}...")
    input_data = load_jsonl_data(args.input)
    print(f"Loaded {len(input_data)} entries")

    # Process answers and extract sympy objects
    results, metrics = process_answers(input_data, args.gold_is_latex)

    # Print comprehensive metrics
    print(f"\n{'='*70}")
    print("EVALUATION METRICS")
    print(f"{'='*70}")
    print(f"\n📊 Basic Metrics:")
    print(f"  Total entries: {metrics['total_count']}")
    print(f"  Correct answers: {metrics['correct_count']}")
    print(f"  Overall accuracy: {metrics['accuracy']:.2%}")

    if 'num_problems' in metrics and metrics['num_problems'] < metrics['total_count']:
        print(f"\n📈 Rollout Statistics:")
        print(f"  Unique problems: {metrics['num_problems']}")
        print(f"  Avg rollouts per problem: {metrics['avg_rollouts_per_problem']:.1f}")

        if 'pass_at_k' in metrics:
            print(f"\n🎯 Pass@k Metrics (estimated):")
            for k in metrics.get('k_values', []):
                key = f'pass@{k}'
                if metrics['pass_at_k'].get(key) is not None:
                    score = metrics['pass_at_k'][key]
                    print(f"  pass@{k}: {score:.2%}")
                else:
                    print(f"  pass@{k}: N/A")

    print(f"{'='*70}\n")

    # Save results to output JSONL
    save_jsonl_data(results, args.output)
    print(f"Results saved to {args.output}")

    # Save metrics to separate JSON file
    metrics_path = args.output.replace('.jsonl', '_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
