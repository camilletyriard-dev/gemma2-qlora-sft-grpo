"""
Dataset loading and formatting for Yoda-style translation and QA SFT.

Datasets:
  - dvgodoy/yoda_sentences  (translation)
  - Synthetic QA (generated in notebook A-Q3)
"""

from typing import Dict
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer
from src.config import YODA_DATASET_NAME, SEED


def load_yoda_dataset(dataset_name: str = YODA_DATASET_NAME) -> DatasetDict:
    """Load the Yoda sentences dataset and create a 90/10 train/val split.

    Returns:
        DatasetDict with 'train' and 'test' splits.
    """
    raw = load_dataset(dataset_name)
    return raw["train"].train_test_split(test_size=0.1, seed=SEED)


def format_yoda_translation_example(
    example: Dict, tokenizer: AutoTokenizer
) -> Dict:
    """Format a Yoda translation pair as a chat-template SFT training string.

    Input field: 'normal'  (standard English sentence)
    Target field: 'yoda'   (Yoda-style translation)

    Returns:
        Dict with 'text' key containing the full formatted training string.
    """
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "user",
             "content": f"Translate to Yoda syntax:\n<text>{example['normal']}</text>"},
            {"role": "model", "content": example["yoda"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": prompt}


def format_qa_yoda_example(example: Dict, tokenizer: AutoTokenizer) -> Dict:
    """Format a synthetic Yoda-style QA pair as a chat-template SFT string.

    Input field:  'question'     (original question from QA dataset)
    Target field: 'yoda_answer'  (Yoda-style answer generated in A-Q3)

    Returns:
        Dict with 'text' key containing the full formatted training string.
    """
    from src.generation import format_prompt_and_answer_qa
    text = format_prompt_and_answer_qa(
        input_text=example["question"],
        answer=example["yoda_answer"],
        tokenizer=tokenizer,
    )
    return {"text": text}