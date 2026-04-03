"""
Dataset loading and formatting for GSM8K mathematical reasoning.

Dataset: openai/gsm8k (HuggingFace)
"""

import re
from typing import Dict, Optional, Tuple

from datasets import DatasetDict, Dataset, load_dataset
from transformers import AutoTokenizer

from src.config import REASONING_DATASET_NAME, SEED


def load_gsm8k_dataset(dataset_name: str = REASONING_DATASET_NAME) -> DatasetDict:
    """Load the GSM8K grade-school math dataset.

    Returns:
        DatasetDict with 'train' and 'test' splits.
    """
    return load_dataset(dataset_name, "main")


def extract_gsm8k_answer_text(answer: str) -> str:
    """Extract the reasoning text from a GSM8K answer (everything before ####).

    Args:
        answer: Full GSM8K answer string containing '#### <number>'.

    Returns:
        Reasoning text only, stripped of the final numeric answer.
    """
    parts = answer.split("####")
    return parts[0].strip() if parts else answer.strip()


def format_gsm8k_for_grpo(example: Dict, tokenizer: AutoTokenizer) -> Dict:
    """Format a GSM8K example as a GRPO training prompt (question only).

    The model must generate a chain-of-thought response ending with #### <number>.

    Args:
        example: Dataset row with 'question' and 'answer' fields.
        tokenizer: Tokenizer with apply_chat_template support.

    Returns:
        Dict with 'prompt' and 'answer' keys.
    """
    content = (
        f"{example['question']}\n\n"
        f"Show your reasoning step by step, "
        f"then write your final answer as: #### <number>"
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return {"prompt": prompt, "answer": example["answer"]}


def prepare_rl_dataset(
    reasoning_ds: Dataset,
    tokenizer: AutoTokenizer,
    n_train: int = 500,
    n_test: int = 200,
    max_answer_words: Optional[int] = 35,
    seed: int = SEED,
) -> Tuple[Dataset, Dataset]:
    """Prepare GSM8K splits for GRPO training.

    Filters for short answers to focus on reasoning chains tractable
    for a 2B-parameter model, then formats prompts for GRPO.

    Args:
        reasoning_ds: The GSM8K training split (DatasetDict['train']).
        tokenizer: Tokenizer used for prompt formatting.
        n_train: Number of training examples to select.
        n_test: Number of test examples to select.
        max_answer_words: Maximum word count for answer filtering. None = no filter.
        seed: Random seed for shuffling.

    Returns:
        Tuple of (train_split, test_split) as HuggingFace Datasets.
    """
    if max_answer_words:
        filtered = reasoning_ds.filter(
            lambda ex: len(ex["answer"].split()) < max_answer_words
        )
        print(f"Answer length filter (<{max_answer_words} words): "
              f"{len(filtered)}/{len(reasoning_ds)} examples kept")
    else:
        filtered = reasoning_ds

    shuffled   = filtered.shuffle(seed=seed)
    train_end  = min(n_train, len(shuffled))
    test_end   = min(n_train + n_test, len(shuffled))
    train_ds   = shuffled.select(range(train_end))
    test_ds    = shuffled.select(range(train_end, test_end))

    def _format(example):
        user_content = (
            f"{example['question']}\n"
            f"Show your reasoning step by step, "
            f"then write your final answer as: #### <number>"
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        )
        return {"prompt": prompt, "ground_truth": example["answer"]}

    train_ds = train_ds.map(_format)
    test_ds  = test_ds.map(_format)

    print(f"RL dataset — Train: {len(train_ds)}, Test: {len(test_ds)}")
    return train_ds, test_ds