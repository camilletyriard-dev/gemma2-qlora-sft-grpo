"""
Inference utilities, prompt formatting, and display helpers for Gemma-2.

All prompt-formatting functions follow the Gemma-2 chat template convention.
Training prompts use add_generation_prompt=False; inference prompts use True.
"""

from typing import List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


# ── Prompt formatting ────────────────────────────────────────────────────────

def format_prompt_yoda(text: str, tokenizer: AutoTokenizer) -> str:
    """Format an English sentence for Yoda-style translation (inference)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Translate to Yoda syntax:\n<text>{text}</text>"}],
        tokenize=False, add_generation_prompt=True,
    )


def format_prompt_gsm8k(question: str, tokenizer: AutoTokenizer) -> str:
    """Format a GSM8K math question for chain-of-thought reasoning (inference)."""
    content = (
        f"{question}\n\nShow your reasoning step by step, "
        f"then write your final answer as: #### <number>"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )


def format_prompt_qa(input_text: str, tokenizer: AutoTokenizer) -> str:
    """Format a QA question for Yoda-style inference (no answer)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": input_text}],
        tokenize=False, add_generation_prompt=True,
    )


def format_prompt_and_answer_qa(
    input_text: str, answer: str, tokenizer: AutoTokenizer
) -> str:
    """Format a QA pair for SFT training (includes expected answer)."""
    return tokenizer.apply_chat_template(
        [
            {"role": "user",      "content": input_text},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False, add_generation_prompt=False,
    )


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_response(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    do_sample: bool = True,
) -> str:
    """Generate a single response from the model given a formatted prompt.

    Returns decoded response string with prompt tokens excluded.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


@torch.no_grad()
def generate_batch_responses(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    batch_size: int = 8,
    temperature: float = 0.1,
) -> List[str]:
    """Generate responses for a list of prompts in batches.

    Processes in batches to avoid OOM errors on large input lists.
    Returns a list of decoded response strings, one per input prompt.
    """
    tokenizer.padding_side = "left"
    all_responses: List[str] = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True).to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=True, pad_token_id=tokenizer.pad_token_id,
        )
        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].shape[0]
            all_responses.append(
                tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
            )

    return all_responses


# ── Text utilities ────────────────────────────────────────────────────────────

def split_sentences(text: str) -> List[str]:
    """Split a text string into individual sentences using punctuation boundaries."""
    import re
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


# ── Display utilities ─────────────────────────────────────────────────────────

def display_examples(
    inputs: List[str],
    before_outputs: List[str],
    after_outputs: List[str],
    labels: Tuple[str, str] = ("Before", "After"),
) -> None:
    """Print before/after model outputs side by side for qualitative inspection."""
    for i, (inp, bef, aft) in enumerate(zip(inputs, before_outputs, after_outputs), 1):
        print(f"{'=' * 70}")
        print(f"Example {i}:")
        print(f"  Input:        {inp}")
        print(f"  {labels[0]:12s}: {bef}")
        print(f"  {labels[1]:12s}: {aft}")
    print(f"{'=' * 70}")