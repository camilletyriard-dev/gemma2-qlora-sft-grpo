# QLoRA Fine-Tuning of Gemma-2 with SFT and GRPO

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Weights-yellow)](https://huggingface.co/camilletyriard/gemma2-qlora-sft-grpo)

Parameter-efficient fine-tuning of **`google/gemma-2-2b-it`** for two tasks:
Yoda-style text generation via **QLoRA + SFT**, and mathematical reasoning
alignment via **GRPO** with a composite reward signal.
Trained on an NVIDIA A100-SXM4-40GB using HuggingFace TRL and PEFT.

---

## Overview

This project investigates two complementary approaches to steering the behaviour
of a 2B-parameter instruction-tuned language model through parameter-efficient
fine-tuning and reinforcement learning from verifiable rewards (RLVR).

**Part A - Style transfer via SFT:**
`gemma-2-2b-it` is fine-tuned with LoRA adapters to translate standard English
into Yoda-style syntax using the `dvgodoy/yoda_sentences` dataset. A synthetic
Yoda-style QA dataset is then generated from `MuskumPillerum/General-Knowledge`
using the trained translator, and a second SFT stage trains the model to answer
any question in Yoda style.

**Part B - Reasoning and style via GRPO:**
GRPO is applied to improve mathematical reasoning on `openai/gsm8k` using a
composite reward signal. A DistilBERT binary classifier, trained to distinguish
Yoda-style from standard English text, provides a differentiable style reward.
Two training strategies are compared: GRPO from the SFT checkpoint (warm start,
correctness + format rewards) and GRPO from the base model (cold start, full
three-component reward).

---

## Methods

### Base Model
- `google/gemma-2-2b-it`, 2.01B parameters, instruction-tuned
- 4-bit NF4 QLoRA quantization (BitsAndBytes) during training
- 8-bit quantization for inference
- Hardware: NVIDIA A100-SXM4-40GB, Google Colab

### Part A: Supervised Fine-Tuning

| Stage | Task | Dataset | Examples |
|---|---|---|---|
| A-1 | Baseline inference | - | Zero-shot Yoda translation |
| A-2 | SFT: English → Yoda | `dvgodoy/yoda_sentences` | 648 train / 72 val |
| A-3 | Synthetic dataset generation | `MuskumPillerum/General-Knowledge` | 500 / 200 / 500 |
| A-4 | SFT: Yoda-style QA | Synthetic (A-3) | 500 train / 200 val |

LoRA is applied to all attention and MLP projection layers.
Training is tracked with Weights & Biases.

### Part B: Reinforcement Learning from Verifiable Rewards

| Stage | Task |
|---|---|
| B-1 | Train DistilBERT binary classifier as style reward model |
| B-2 | Design and validate composite reward function |
| B-3 | GRPO from SFT checkpoint — correctness + format rewards |
| B-4 | GRPO from base model — correctness + format + style rewards |
| B-5 | Quantitative and qualitative comparison |

### Reward Function

| Component | Implementation | Range |
|---|---|---|
| Correctness | Graduated exact-match: 1.0 for `#### n`, 0.5 for "answer is n" | [0, 1] |
| Format | Rule-based: presence of `#### <number>` marker | {0, 1} |
| Style | DistilBERT P(Yoda), trained on GSM8K English/Yoda pairs | [0, 1] |
| **Total** | Linear sum | [0, 3] |

### Dataset Quality Filtering (A-3)

A three-tier filter ensures Yoda-translatability of QA pairs:

1. **Tier 1: Regex**: word count bounds (4–15 per sentence), character
   whitelist, no questions in answers, no translation tasks, number density
2. **Tier 2: Structural**: no duplicate sentences, no list/enumeration patterns  
3. **Tier 3: NLP**: passive voice rate < 50% per sentence (spaCy)

---

## Results

### Part A — Yoda Translation and QA

| Model | Example output |
|---|---|
| Base Gemma-2-2b-it | *"Paris, art galleries, renowned, it is."* |
| + SFT translation | *[consistent Yoda syntax, correct OSV word order]* |
| + SFT Yoda QA | *[answers questions in Yoda style across domains]* |

### Part B — GRPO Comparison

| Strategy | Correctness | Format Compliance | Style Score | Total Reward |
|---|---|---|---|---|
| SFT + GRPO (B-Q3) | 0.667 | 0.833 | — | — |
| Base + GRPO (B-Q4) | 0.667 | 1.000 | 0.379 | — |

B-Q4 achieves perfect format compliance; B-Q3 shows higher early-training
stability due to the SFT warm start. Full per-example analysis is in
`notebooks/experiments.ipynb` Section B-5.

---

## Repository Structure

```text

├── notebooks/
│   └── experiments.ipynb          # Complete experiment notebook (A + B)
│
├── src/                           # Importable Python library
│   ├── config.py                  # Paths, model names, global seed
│   ├── model.py                   # QLoRA loading, PEFT utilities
│   ├── generation.py              # Prompt formatting, inference, display
│   ├── data/
│   │   ├── yoda.py                # Yoda dataset loading and formatting
│   │   ├── gsm8k.py               # GSM8K loading and RL dataset preparation
│   │   └── qa_filter.py           # Multi-tier QA quality filter
│   ├── training/
│   │   ├── sft.py                 # SFTTrainer + LoRA config
│   │   ├── grpo.py                # GRPOTrainer builder
│   │   └── classifier.py          # DistilBERT classifier training
│   ├── rewards/
│   │   ├── correctness.py         # Graduated numeric match reward
│   │   ├── format.py              # Answer format compliance reward
│   │   └── style.py               # Classifier-based style reward
│   └── evaluation/
│       ├── metrics.py             # Classifier eval, before/after scoring
│       └── plotting.py            # Training curves, experiment comparison
│
├── scripts/
│   ├── train_sft.py               # CLI: run SFT
│   ├── train_grpo.py              # CLI: run GRPO
│   └── inference.py               # CLI: inference on new text
│
├── results/
│   ├── sft/metrics.json
│   └── grpo/metrics.json
│
└── checkpoints/                   # Empty — adapters hosted on Hugging Face

```


---

## Setup

### 1. Clone
```bash
git clone https://github.com/camilletyriard-dev/gemma2-qlora-sft-grpo.git
cd gemma2-qlora-sft-grpo
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> Requires a GPU with at least **16 GB VRAM** for 4-bit inference.
> Full training was conducted on an NVIDIA A100-SXM4-40GB (~12–24 GPU hours
> inclusive of hyperparameter search).

### 3. Set credentials
```bash
cp .env.example .env
# Edit .env — never commit this file
```

Or export directly:
```bash
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key
export CHECKPOINT_DIR=/path/to/checkpoints   # or Google Drive path in Colab
```

### 4. Download LoRA adapters

Pre-trained weights are hosted on Hugging Face. Download and place adapter
folders in `checkpoints/`.

🤗 **[huggingface.co/camilletyriard/gemma2-qlora-sft-grpo](https://huggingface.co/camilletyriard/gemma2-qlora-sft-grpo)**

| Adapter | Description |
|---|---|
| `sft_yoda/` | A-2: English → Yoda translator |
| `sft_yoda_answ/` | A-4: Yoda-style QA answerer |
| `rl_yoda_answ_from_sft/` | B-3: GRPO warm start (best config) |
| `rl_yoda_answ_from_base/` | B-4: GRPO cold start (best config) |
| `classifier_yoda/` | B-1: DistilBERT style classifier |

### 5. Run the notebook
```bash
jupyter lab notebooks/experiments.ipynb
```

The notebook is structured as a linear pipeline: run top-to-bottom.
Each training stage has a `skip_training = True` guard — set to `False`
to retrain from scratch.

### 6. Run inference from the command line
```bash
# Yoda-style translation
python scripts/inference.py \
  --adapter checkpoints/sft_yoda \
  --task yoda \
  --text "The stars are bright tonight."

# Mathematical reasoning
python scripts/inference.py \
  --adapter checkpoints/rl_yoda_answ_from_sft \
  --task gsm8k \
  --text "Janet has 3 apples and gives 1 to Bob. How many remain?"
```

---

## Key Design Decisions

**QLoRA over full fine-tuning.** Gemma-2-2b-it at full precision requires ~16 GB
VRAM. 4-bit NF4 quantization reduces this to ~3 GB, enabling training on a single
GPU without meaningful performance degradation on downstream tasks.

**GRPO over PPO.** GRPO eliminates the value network required by PPO, reducing
memory overhead by ~50% while producing comparable results on reasoning tasks.
Rewards are computed relative to a group of sampled responses rather than an
absolute learned baseline, making the training signal more stable at small batch
sizes.

**Learned style reward over rule-based detection.** A rule-based Yoda detector is
brittle to paraphrase. A DistilBERT classifier trained on balanced English/Yoda
pairs provides a differentiable reward signal that captures stylistic nuance beyond
simple OSV word-order patterns.

**Graduated correctness reward.** The standard binary exact-match reward (1 or 0)
produces a sparse signal for a 2B model early in training. A 0.5 partial-credit
score for "the answer is N" formulations provides a denser gradient early in GRPO
training, accelerating convergence toward the required `#### <number>` format.

**Three-tier QA filter.** Raw QA pairs from general-knowledge datasets often
contain sentences that are too short, heavily passive, or structurally complex to
produce coherent Yoda translations. The filter improves translation quality
substantially and reduces noise in the SFT training signal for A-4.

---

## Limitations

- Results are reported for a single random seed (42); variance across seeds is
  not quantified.
- The style classifier is trained on the same distribution as the SFT data,
  which may inflate style reward estimates during GRPO evaluation.
- Google Colab session limits constrained the hyperparameter search to a small
  number of configurations per stage.
- GSM8K evaluation covers a limited held-out sample; reported rewards are
  indicative rather than benchmark-scale.
- The SFT answerer was trained on general-knowledge QA; applying it to GSM8K
  mathematical reasoning introduces a domain shift that reduces initial
  correctness scores before GRPO compensates.

---

## Contributors

Developed at University College London (2025–2026):
Camille Tyriard, Ana Quintero, Dunia Tornila, Daniel Huencho,
Jonathan Bell, Nicolas Tobo.

