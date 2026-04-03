"""
Correctness reward for GRPO: exact numeric match against GSM8K ground truth.

Two interfaces:
  - correctness_reward(prompts, completions, **kwargs) -> List[float]
    GRPO batch interface (used by GRPOTrainer)
  - correctness_reward_single(response, ground_truth) -> float
    Single-sample interface (used for evaluation display)
"""

import re
from typing import List, Optional


def extract_gsm8k_final_answer(text: str) -> Optional[str]:
    """Extract the numeric answer following the '####' marker.

    Args:
        text: Model response or ground-truth string.

    Returns:
        Extracted numeric string (commas stripped), or None if not found.
    """
    match = re.search(r"####\s*([\d,.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


def _check_numeric_match(val_str: str, gt_num: float) -> bool:
    """Return True if val_str parses to a number matching gt_num."""
    try:
        clean = re.sub(r"[^\d.\+\-]", "", val_str)
        return abs(float(clean) - gt_num) < 1e-5
    except (ValueError, AttributeError):
        return False


def correctness_reward_single(response: str, ground_truth: str) -> float:
    """Score a single response against a ground-truth answer.

    Implements graduated extraction:
      1.0 — exact match via '#### <number>' marker (instructed format)
      0.5 — match via 'the answer is <number>' pattern (partial credit)
      0.0 — no match found

    Args:
        response: Model-generated response string.
        ground_truth: GSM8K ground-truth answer string (contains #### <n>).

    Returns:
        Reward score in {0.0, 0.5, 1.0}.
    """
    gt_answer = extract_gsm8k_final_answer(ground_truth)
    if not gt_answer:
        return 0.0

    try:
        gt_clean = re.sub(r"[^\d.\+\-]", "", gt_answer)
        gt_num = float(gt_clean)
    except ValueError:
        return 0.0

    num_pattern = r"([+\-]?\s*[\$£€¥]?\s*[\d,]+\.?\d*)"

    # Priority 1: #### <number> — full credit
    match = re.search(r"####\s*" + num_pattern, response)
    if match and _check_numeric_match(match.group(1), gt_num):
        return 1.0

    # Priority 2: "the answer is <number>" — partial credit
    match = re.search(
        r"(?:the\s+answer\s+is|the\s+final\s+answer\s+is|answer\s*[:=])\s*" + num_pattern,
        response, re.IGNORECASE,
    )
    if match and _check_numeric_match(match.group(1), gt_num):
        return 0.5

    return 0.0


def correctness_reward(
    prompts: List[str], completions: List[str], **kwargs
) -> List[float]:
    """GRPO batch reward: correctness score for each completion.

    Follows the GRPOTrainer reward function interface.

    Args:
        prompts: List of input prompt strings (unused, required by interface).
        completions: List of model-generated response strings.
        **kwargs: Must contain 'answer' or 'ground_truth' — list of GT strings.

    Returns:
        List of reward scores via correctness_reward_single.
    """
    ground_truths = kwargs.get("answer", kwargs.get("ground_truth",
                               [""] * len(completions)))
    return [
        correctness_reward_single(c, gt)
        for c, gt in zip(completions, ground_truths)
    ]