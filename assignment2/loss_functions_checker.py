"""Reference checks for the student cross-entropy loss implementation.

Run this module as a script to compare `compute_loss_from_logits` against the
reference implementation on a few deterministic examples:

    python loss_functions_checker.py
"""

from __future__ import annotations

import os
import pickle
import sys
from typing import Dict, Iterable, List, Tuple

import torch

from loss_functions import compute_loss_from_logits
from transformers.modeling_outputs import CausalLMOutputWithPast


class LossCheckFailure(Exception):
    """Raised when the student implementation diverges from the reference."""


def _load_reference_answers() -> Dict[int, float]:
    """Load reference answers from pickle file.

    Returns:
        Dictionary mapping case index to reference loss value.

    Raises:
        FileNotFoundError: If reference_answers.pkl does not exist.
    """
    reference_file = "reference_answers.pkl"
    if not os.path.exists(reference_file):
        raise FileNotFoundError(
            f"Reference answers file '{reference_file}' not found. "
            "Please run 'python generate_reference_answers.py' first."
        )

    with open(reference_file, "rb") as f:
        return pickle.load(f)


def _generate_test_cases() -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
    """Generate test cases with corresponding num_items_in_batch.

    Returns:
        List of tuples (logits, labels, num_items_in_batch).
    """
    test_cases = []

    # Case 1: No ignored tokens
    torch.manual_seed(0)
    logits_a = torch.randn(2, 3, 5, dtype=torch.float32)
    labels_a = torch.tensor(
        [
            [0, 2, 4],
            [1, 3, 0],
        ],
        dtype=torch.long,
    )
    num_items_a = 4  # 2*3 - 2 (shift by 1) = 4 valid tokens
    test_cases.append((logits_a, labels_a, num_items_a))

    # Case 2: With ignored tokens (-100)
    torch.manual_seed(1)
    logits_b = torch.randn(1, 4, 7, dtype=torch.float32)
    labels_b = torch.tensor([[2, -100, 6, -100]], dtype=torch.long)
    num_items_b = 2  # Only 2 valid tokens after masking
    test_cases.append((logits_b, labels_b, num_items_b))

    # Case 3: Multiple ignored tokens in different positions
    torch.manual_seed(42)
    logits_c = torch.randn(3, 2, 11, dtype=torch.float32)
    labels_c = torch.tensor(
        [
            [0, 1],
            [4, -100],
            [-100, 7],
        ],
        dtype=torch.long,
    )
    num_items_c = 4  # 4 valid tokens after masking
    test_cases.append((logits_c, labels_c, num_items_c))

    return test_cases


def _evaluate_case(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: int,
    reference_loss_value: float,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> torch.Tensor:
    """Evaluate a single test case against reference answer.

    Args:
        logits: Model logits with shape [batch_size, seq_len, vocab_size].
        labels: Ground truth labels with shape [batch_size, seq_len].
        num_items_in_batch: Number of valid items in batch for normalization.
        reference_loss_value: Reference loss value from pickle file.
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.

    Returns:
        Student loss tensor.

    Raises:
        LossCheckFailure: If student loss diverges from reference.
    """
    # Create CausalLMOutputWithPast object as expected by compute_loss_from_logits
    outputs = CausalLMOutputWithPast(logits=logits)

    student_loss = compute_loss_from_logits(
        outputs, labels, num_items_in_batch=num_items_in_batch
    )

    student_loss_value = student_loss.detach().cpu().item()
    reference_tensor = torch.tensor(reference_loss_value, dtype=student_loss.dtype)

    if not torch.allclose(student_loss, reference_tensor, atol=atol, rtol=rtol):
        raise LossCheckFailure(
            f"Loss mismatch.\n"
            f"Student:   {student_loss_value:.8f}\n"
            f"Reference: {reference_loss_value:.8f}\n"
            f"Absolute difference: {abs(student_loss_value - reference_loss_value):.8f}\n"
            f"atol={atol}, rtol={rtol}"
        )

    if torch.isnan(student_loss):
        raise LossCheckFailure("Student loss produced NaN.")

    return student_loss


def run_loss_checks(verbose: bool = True) -> None:
    """Runs reference parity checks for the student loss implementation.

    Args:
        verbose: Whether to print progress information.

    Raises:
        RuntimeError: If reference answers file is missing or compute_loss_from_logits is not implemented.
        LossCheckFailure: If student implementation diverges from reference.
    """
    # Load reference answers
    try:
        reference_answers = _load_reference_answers()
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc

    # Generate test cases
    test_cases = _generate_test_cases()

    if len(test_cases) != len(reference_answers):
        raise RuntimeError(
            f"Mismatch between number of test cases ({len(test_cases)}) "
            f"and reference answers ({len(reference_answers)}). "
            "Please regenerate reference answers."
        )

    # Run checks
    for case_index, (logits, labels, num_items) in enumerate(test_cases, start=1):
        reference_loss_value = reference_answers[case_index]

        try:
            student_loss = _evaluate_case(
                logits, labels, num_items, reference_loss_value
            )
        except NotImplementedError as exc:
            raise RuntimeError(
                "Unable to run loss checks because `compute_loss_from_logits` is not implemented."
            ) from exc

        if verbose:
            print(
                f"[case {case_index}] passed "
                f"(num_items={num_items}): "
                f"loss={student_loss.detach().cpu().item():.8f}"
            )


def main(argv: Iterable[str] | None = None) -> int:
    """Main entry point for the checker script."""
    del argv  # Unused.
    try:
        run_loss_checks()
        print("\nAll checks passed!")
    except LossCheckFailure as failure:
        print(f"Loss parity check failed:\n{failure}", file=sys.stderr)
        return 1
    except RuntimeError as runtime_exc:
        print(runtime_exc, file=sys.stderr)
        return 1
    except NotImplementedError:
        print(
            "compute_loss_from_logits is not implemented yet; "
            "complete the assignment before running this checker.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
