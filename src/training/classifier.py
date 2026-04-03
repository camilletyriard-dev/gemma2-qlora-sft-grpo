"""
DistilBERT binary classifier training for Yoda-style reward model (B-Q1).

Builds a balanced English/Yoda dataset by translating GSM8K answers with
the trained SFT translator, then fine-tunes DistilBERT as a binary classifier.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.config import CLASSIFIER_MODEL_NAME, SEED


def build_classifier_dataset(
    reasoning_ds,
    translator_model,
    translator_tokenizer,
    n_samples: int = 1000,
    batch_size: int = 32,
    cache_path: Optional[Path] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Build a balanced English/Yoda classifier dataset from GSM8K answers.

    For each reasoning answer:
      - original text  → label 0 (English)
      - Yoda translation → label 1 (Yoda)

    Translations are generated sentence-by-sentence using the trained SFT
    translator from Part A, then reassembled into full-answer strings.

    Args:
        reasoning_ds: GSM8K training split (HuggingFace Dataset).
        translator_model: Trained SFT translation model (PEFT or merged).
        translator_tokenizer: Corresponding tokenizer.
        n_samples: Number of reasoning answers to use (produces 2*n samples).
        batch_size: Batch size for batch translation.
        cache_path: If provided, save/load splits here to avoid recomputation.

    Returns:
        Tuple of (train_ds, val_ds, test_ds) as HuggingFace Datasets.
    """
    from src.model import checkpoint_exists
    from src.generation import generate_batch_responses, split_sentences, format_prompt_yoda
    from src.data.gsm8k import extract_gsm8k_answer_text
    from src.config import SEED

    # ── Check cache ──────────────────────────────────────────────────────────
    if cache_path and checkpoint_exists(cache_path / "train"):
        print("Loading cached classifier dataset...")
        train = Dataset.load_from_disk(str(cache_path / "train"))
        val   = Dataset.load_from_disk(str(cache_path / "val"))
        test  = Dataset.load_from_disk(str(cache_path / "test"))
        return train, val, test

    # ── Sample and filter reasoning answers ──────────────────────────────────
    subset = reasoning_ds.shuffle(seed=SEED).select(range(min(n_samples, len(reasoning_ds))))
    english_texts = [extract_gsm8k_answer_text(ex["answer"]) for ex in subset]
    english_texts = [t for t in english_texts if len(t.split()) >= 10]
    print(f"Kept {len(english_texts)} answers with >= 10 words")

    # ── Translate sentence-by-sentence ───────────────────────────────────────
    all_sentences, boundaries = [], []
    for text in english_texts:
        sents = split_sentences(text)
        start = len(all_sentences)
        all_sentences.extend(sents)
        boundaries.append((start, len(all_sentences)))

    prompts = [format_prompt_yoda(s, translator_tokenizer) for s in all_sentences]
    print(f"Translating {len(prompts)} sentences...")
    translated = generate_batch_responses(
        translator_model, translator_tokenizer, prompts,
        batch_size=batch_size, max_new_tokens=128,
    )
    yoda_texts = [" ".join(translated[s:e]) for s, e in boundaries]

    # ── Build balanced dataset ────────────────────────────────────────────────
    texts  = english_texts + yoda_texts
    labels = [0] * len(english_texts) + [1] * len(yoda_texts)

    indices = list(range(len(texts)))
    tr_idx, te_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=SEED)
    tr_idx, va_idx = train_test_split(
        tr_idx,
        test_size=0.1,
        stratify=[labels[i] for i in tr_idx],
        random_state=SEED,
    )

    def _make_ds(idx):
        return Dataset.from_dict({
            "text":  [texts[i] for i in idx],
            "label": [labels[i] for i in idx],
        })

    train_ds = _make_ds(tr_idx)
    val_ds   = _make_ds(va_idx)
    test_ds  = _make_ds(te_idx)

    print(f"Classifier dataset — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
        train_ds.save_to_disk(str(cache_path / "train"))
        val_ds.save_to_disk(str(cache_path / "val"))
        test_ds.save_to_disk(str(cache_path / "test"))
        print(f"Cached to {cache_path}")

    return train_ds, val_ds, test_ds


def build_classifier_trainer(
    model,
    tokenizer,
    train_ds: Dataset,
    eval_ds: Dataset,
    output_dir: Path,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 256,
) -> Trainer:
    """Build and return a configured Trainer for DistilBERT classification.

    Args:
        model: AutoModelForSequenceClassification instance.
        tokenizer: Corresponding tokenizer.
        train_ds: Training split with 'text' and 'label' columns.
        eval_ds: Validation split.
        output_dir: Checkpoint directory.
        num_train_epochs: Training epochs.
        per_device_train_batch_size: Batch size.
        learning_rate: Initial learning rate.
        max_length: Maximum token length.

    Returns:
        Configured Trainer ready to call .train().
    """
    from sklearn.metrics import accuracy_score

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], truncation=True,
            max_length=max_length, padding=False,
        )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds)}

    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_tok  = eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=64,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        bf16=True,
        logging_steps=10,
        seed=SEED,
        report_to="wandb",
    )

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )