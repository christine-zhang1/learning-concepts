# Learning Concepts, Not Tokens: Self-Supervised Semantic Alignment for Language Models

Fine-tuning a causal language model with a **concept loss** that rewards placing
probability mass on a *set* of interchangeable tokens (a content word
plus its contextual synonyms) at each target position, rather than only on the
single ground-truth next token. This repository contains the full pipeline:
data collection, baseline data processing, training, and evaluation.

## Layout

```
concept-training/
├── conceptlib/                 # shared library (importable package)
│   ├── config.py               # TrainingConfig dataclass
│   ├── paths.py                # env-configurable roots + path builders
│   ├── csv_utils.py            # long-format results-CSV upsert helpers
│   └── eval_utils.py           # model loading + bootstrap/timing helpers
├── data/                       # data collection + baseline processing
│   ├── get_content_words.py     # 1. pick single-token content words
│   ├── embedding_synonyms.py   # 2a. embedding-based synonyms
│   ├── llm_synonyms.py         # 2b. LLM-prompted synonyms
│   ├── merge_synonym_parts.py  # 3. merge shards -> train/val/test splits
│   ├── augment_synonyms.py     # baseline: text augmentation by synonym swap
│   └── randomize_synonyms.py   # baseline: randomized-synonym control set
├── train.py                    # concept-loss fine-tuning
├── eval/
│   ├── eval_mteb.py            # MTEB embedding benchmark
│   ├── eval_ntp.py             # next-token prediction (all positions)
│   ├── eval_ntp_content_words.py   # NTP at content-word positions
│   └── eval_lm_harness.py      # lm-evaluation-harness benchmark sweep
├── pyproject.toml
└── requirements.txt
```

## Setup

```bash
pip install -e .
python -m spacy download en_core_web_sm
```

`pip install -e .` makes `conceptlib` importable. The scripts also insert the
project root onto `sys.path` themselves, so they run without installation too.

## Configuring paths

Three environment variables control where data, checkpoints, and results live 
(defaults are relative to the current directory):

| Variable                  | Default         | Holds                                   |
|---------------------------|-----------------|-----------------------------------------|
| `CONCEPT_DATA_ROOT`       | `./datasets`    | preprocessed dataset JSONL files        |
| `CONCEPT_CHECKPOINT_ROOT` | `./checkpoints` | trained LoRA adapter checkpoints        |
| `CONCEPT_RESULTS_ROOT`    | `./results`     | evaluation result CSVs                  |

```bash
export CONCEPT_DATA_ROOT=/path/to/data
export CONCEPT_CHECKPOINT_ROOT=/path/to/checkpoints
export CONCEPT_RESULTS_ROOT=/path/to/results
```

Within `CONCEPT_DATA_ROOT`, files follow `{dataset}/{model}/...` where
`{model}` is the lowercased final segment of the HF model name (e.g. `qwen3-4b-base`),
and `{dataset_type}` is `embedding` or `prompting`.

## Pipeline

### 1. Data collection

```bash
# 1. Extract single-token content words -> {data}/{dataset}/{model}/combined.jsonl
python data/get_content_words.py --model meta-llama/Llama-3.2-1B --dataset c4

# 2a. Embedding-based synonyms (also writes the top-k token pool used by 2b)
python data/embedding_synonyms.py c4 --start 0 --end 10000 --model meta-llama/Llama-3.2-1B

# 2b. LLM-prompted synonyms (consumes the top-k pool from 2a)
python data/llm_synonyms.py c4 --start 0 --end 10000 --model meta-llama/Llama-3.2-1B

# 3. Merge the per-shard synonyms_{start}_{end}.jsonl files into
#    synonyms_{train,val,test}.jsonl for every dataset/model/{embedding,prompting}.
#    Audits each directory first; pass --force to split only the ready ones.
python data/merge_synonym_parts.py --train-size 8000 --val-size 1000 --test-size 1000
```

Steps 2a/2b write per-shard files under `embedding/` and `prompting/`; step 3
merges them into `synonyms_train.jsonl`, `synonyms_val.jsonl`, and
`synonyms_test.jsonl` per `{dataset}/{model}/{dataset_type}/` — the files
training and evaluation consume.

### 2. Baseline data processing (optional)

```bash
# Text-augmentation baseline (synonym swaps baked into the training text)
python data/augment_synonyms.py --num-augmentations 4

# Randomized-synonym control set. --split chooses which split(s) to randomize:
#   train -> synonyms_train_randomized.jsonl (consumed by train.py --randomized-synonyms)
#   test  -> synonyms_test_randomized.jsonl  (default)
python data/randomize_synonyms.py --split train
python data/randomize_synonyms.py --split test
```

### 3. Training

```bash
python train.py \
    --model-name Qwen/Qwen3-4B-Base \
    --dataset c4 \
    --dataset-type embedding \
    --concept-loss-weight 0.5 \
    --target-mass 0.6
```

Checkpoints are written under `CONCEPT_CHECKPOINT_ROOT` at the path built by
`conceptlib.paths.adapter_dir` — the same builder the eval scripts use to locate
adapters, so training output and evaluation lookup never drift. Ablation flags
(`--randomized-synonyms`, `--use-weighted-ce`, `--max-train-samples`,
`--use-data-augmentation`, `--max-content-words`) each
select a distinct checkpoint subdirectory.

### 4. Evaluation

Each eval script sweeps the adapter weights for a configuration and upserts a
long-format CSV under `CONCEPT_RESULTS_ROOT`.

```bash
# MTEB embedding benchmark (STS preset and the short-text preset)
python eval/eval_mteb.py --base-model Qwen/Qwen3-4B-Base --dataset c4 \
    --dataset-type embedding --tasks sts
python eval/eval_mteb.py --base-model Qwen/Qwen3-4B-Base --dataset c4 \
    --dataset-type embedding --tasks short

# Next-token prediction over all positions
python eval/eval_ntp.py --base-model Qwen/Qwen3-4B-Base \
    --model-training-dataset c4 --evaluation-dataset c4 --dataset-type embedding

# Next-token prediction at content-word positions
python eval/eval_ntp_content_words.py --base-model Qwen/Qwen3-4B-Base \
    --model-training-dataset c4 --evaluation-dataset c4 --dataset-type embedding

# lm-evaluation-harness benchmarks (ARC-Challenge/Easy, OpenBookQA,
# HellaSwag, PIQA, WinoGrande). Shells out to the `lm_eval` CLI per adapter.
python eval/eval_lm_harness.py --base-model Qwen/Qwen3-4B-Base \
    --model-training-dataset c4 --dataset-type embedding
```

`eval_lm_harness.py` runs the six likelihood-scored benchmarks above by default;
pass `--tasks-suite custom --tasks ...` for an arbitrary lm-eval task list. It
requires the `lm-eval` package (in `requirements.txt`).

`eval_mteb.py` exposes two presets via `--tasks` (`sts`, `short`) or a
comma-separated task list. `STS_PRESET` and `MTEB_SHORT_PRESET` (the `short`
preset) are defined at the top of the script.

## Citation

We will be presenting at EMNLP 2026 in Budapest!

If you find this repository useful, please cite:

```bibtex
@misc{zhang2026learningconceptstokensselfsupervised,
      title={Learning Concepts, Not Tokens: Self-Supervised Semantic Alignment for Language Models}, 
      author={Christine Zhang and Dan Jurafsky and Chen Shani},
      year={2026},
      eprint={2603.29123},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.29123}, 
}
```
