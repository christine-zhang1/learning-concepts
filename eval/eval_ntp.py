"""Evaluate next-token-prediction quality on the held-out synonyms_test split.

Reads `input_sequence` from
  {CONCEPT_DATA_ROOT}/{evaluation_dataset}/{model}/{dataset_type}/synonyms_test.jsonl
(the same file used by eval_ntp_content_words.py — synonym fields are
ignored here since this script scores every position).

Computes, per adapter weight, averaged over sequences:
  - token-level accuracy
  - average confidence
  - negative log-likelihood
  - perplexity
plus paired-bootstrap CIs across adapters.
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import time
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from conceptlib import csv_utils as ecu
from conceptlib import paths
from conceptlib.eval_utils import (
    load_model_and_tokenizer,
    paired_bootstrap_cis,
    format_elapsed,
    free_model_memory,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


ADAPTER_WEIGHTS = ["base", "0.0", "0.25", "0.5", "0.75", "1.0"]


def build_adapter_dir(args, adapter_weight, ce_concept_token_weight=None):
    """Construct the absolute adapter directory matching the training layout
    (conceptlib.paths.adapter_dir). Returns None for the bare base model."""
    if adapter_weight == "base":
        return None
    return paths.adapter_dir(
        args.model_training_dataset, args.base_model, args.dataset_type,
        max_content_words=args.max_content_words,
        randomized_synonyms=args.randomized_synonyms,
        use_weighted_ce=args.use_weighted_ce,
        ce_concept_token_weight=ce_concept_token_weight,
        max_train_samples=args.max_train_samples,
        use_data_augmentation=args.use_data_augmentation,
        num_train_epochs=args.num_train_epochs,
        concept_loss_weight=adapter_weight,
    )


def _tokenize_texts(texts, tokenizer, max_length):
    """Pre-tokenize texts once. Returns list of {'input_ids', 'attention_mask'} dicts
    of 1D tensors so __getitem__ is a free lookup across adapter sweeps."""
    out = []
    for text in tqdm(texts, desc="Pre-tokenizing"):
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out.append({
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        })
    return out


class JsonlDataset(Dataset):
    """Pre-tokenized dataset that reads input_sequence from a JSONL file."""

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ):
        texts = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Loading {data_path}"):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                text = item.get("input_sequence", "")
                if not isinstance(text, str) or not text.strip():
                    continue
                texts.append(text.strip())

        logger.info(f"Loaded {len(texts)} samples from {data_path}")
        self.encodings = _tokenize_texts(texts, tokenizer, max_length)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def _collate_fn(batch):
    """Dynamically pad to the longest sequence in the batch."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    padded_input_ids = []
    padded_attention_mask = []
    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len
        padded_input_ids.append(torch.cat([b["input_ids"], b["input_ids"].new_zeros(pad_len)]))
        padded_attention_mask.append(torch.cat([b["attention_mask"], b["attention_mask"].new_zeros(pad_len)]))
    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
    }


# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate(model, dataset, batch_size=8, device="cuda", return_per_example=False):
    """Evaluate NTP over all token positions per sequence (then average per sequence).
    If return_per_example=True, also return per-example lists for paired bootstrap."""
    model.eval()
    model.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_fn,
    )

    all_confidences = []
    all_correctness = []
    all_nll = []

    for batch in tqdm(dataloader, desc="Evaluating"):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Vectorized: compute per-sequence mean NTP metrics over all valid positions.
        # Position t is valid when attention_mask[:,t]==1 AND attention_mask[:,t-1]==1.
        # logits[:,t-1,:] predicts the token at position t.
        valid = (attention_mask[:, 1:] & attention_mask[:, :-1]).bool()  # [B, seq_len-1]
        shift_logits  = logits[:, :-1, :]
        shift_targets = input_ids[:, 1:]

        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        probs     = log_probs.exp()

        true_log_probs = log_probs.gather(2, shift_targets.unsqueeze(2)).squeeze(2)
        max_probs      = probs.max(dim=-1).values
        pred_tokens    = probs.argmax(dim=-1)
        correct        = (pred_tokens == shift_targets).float()

        true_log_probs = true_log_probs.masked_fill(~valid, 0.0)
        max_probs      = max_probs.masked_fill(~valid, 0.0)
        correct        = correct.masked_fill(~valid, 0.0)

        valid_counts = valid.float().sum(dim=1)
        has_valid    = valid_counts > 0
        safe_counts  = valid_counts.clamp(min=1)

        seq_conf    = max_probs.sum(dim=1) / safe_counts
        seq_correct = correct.sum(dim=1) / safe_counts
        seq_nll     = -true_log_probs.sum(dim=1) / safe_counts

        all_confidences.extend(seq_conf[has_valid].tolist())
        all_correctness.extend(seq_correct[has_valid].tolist())
        all_nll.extend(seq_nll[has_valid].tolist())

    total_examples = len(all_confidences)
    avg_conf = np.mean(all_confidences)
    accuracy = np.mean(all_correctness)
    avg_nll = np.mean(all_nll)
    perplexity = np.exp(avg_nll)

    out = {
        "accuracy": float(accuracy),
        "avg_confidence": float(avg_conf),
        "nll": float(avg_nll),
        "perplexity": float(perplexity),
        "num_examples": total_examples,
    }

    if return_per_example:
        out["_confidences"] = np.asarray(all_confidences)
        out["_correctness"] = np.asarray(all_correctness)
        out["_nll"] = np.asarray(all_nll)
    return out


def run_eval_for_adapter(adapter_path: str, dataset, args, return_per_example=False, ce_concept_token_weight=None):
    """Load model for this adapter, run NTP eval, return metrics dict. Caller must free model after."""
    full_path = build_adapter_dir(args, adapter_path, ce_concept_token_weight)
    model, _ = load_model_and_tokenizer(
        adapter_full_path=full_path,
        base_model_name=args.base_model,
    )
    metrics = evaluate(
        model,
        dataset,
        batch_size=args.batch_size,
        device=args.device,
        return_per_example=return_per_example,
    )
    return metrics


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--model-training-dataset", type=str, required=True)
    parser.add_argument("--evaluation-dataset", type=str, required=True,
                        help="Eval dataset name (c4, openwebtext); selects the synonyms_test.jsonl path.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--csv-output",
        type=str,
        default=ecu.results_csv("ntp"),
        help=f"Long-format CSV to upsert results into (default: {ecu.results_csv('ntp')}).",
    )
    parser.add_argument("--randomized-synonyms", action="store_true")
    parser.add_argument("--max-content-words", type=str, default=None, help="Choose between all, half, quarter, or one (default: all)")
    parser.add_argument("--use-weighted-ce", action="store_true")
    parser.add_argument("--dataset-type", type=str, required=True, choices=["embedding", "prompting"],
                        help="Dataset type used during training; part of the adapter path.")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="If set, look under max-train-samples-{N}/ subdir.")
    parser.add_argument("--use-data-augmentation", action="store_true",
                        help="If set, look under aug_{N}epochs/ subdir.")
    parser.add_argument("--num-train-epochs", type=int, default=None,
                        help="Epoch count used to name the aug subdir (required with --use-data-augmentation).")
    ecu.add_ablation_arg(parser)

    args = parser.parse_args()

    if args.use_data_augmentation and args.num_train_epochs is None:
        parser.error("--use-data-augmentation requires --num-train-epochs")

    adapter_weights = list(ADAPTER_WEIGHTS)
    if (args.randomized_synonyms
            or (args.max_content_words and args.max_content_words != "all")):
        # Prepend "base" so paired-bootstrap diff CIs are computed vs. base
        # instead of vs. the first NONZERO adapter.
        adapter_weights = ["base", "0.25", "0.5", "0.75", "1.0"]
    elif args.use_data_augmentation:
        adapter_weights = ["base", "0.0"]

    # Get tokenizer and dataset once (tokenizer is same for all adapters)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    eval_dataset_label = args.evaluation_dataset
    data_path = os.path.join(
        paths.dataset_dir(args.evaluation_dataset, args.base_model, args.dataset_type),
        "synonyms_test.jsonl",
    )
    print(f"Loading {data_path} for {args.base_model}...")
    dataset = JsonlDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    print(f"Dataset: {len(dataset)} samples")

    per_model_metrics = []  # list of dicts with _confidences, _correctness, _nll
    metrics_per_adapter = []  # full metrics dict per adapter, same order as adapter_weights

    if args.use_weighted_ce:
        # ce_weight=1.0 isn't trained; prepend base model so wce runs get
        # paired diff CIs against base rather than against an untrained ce=1.0.
        ce_weights = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        iter_configs = [("base", None)] + [("0.0", w) for w in ce_weights]
    else:
        iter_configs = [(ap, None) for ap in adapter_weights]

    for adapter_path, ce_weight in iter_configs:
        print(f"\n--- Adapter: {adapter_path}{f', ce_weight={ce_weight}' if ce_weight is not None else ''} ---")
        _t_start = time.time()
        try:
            print(f"Running evaluation for adapter: {adapter_path}")
            metrics = run_eval_for_adapter(
                adapter_path, dataset, args, return_per_example=True, ce_concept_token_weight=ce_weight
            )
            per_model_metrics.append({
                "_confidences": metrics["_confidences"],
                "_correctness": metrics["_correctness"],
                "_nll": metrics["_nll"],
            })
            metrics_per_adapter.append(metrics)
            print(f"  accuracy={metrics['accuracy']:.4f}  "
                  f"avg_conf={metrics['avg_confidence']:.4f}  "
                  f"nll={metrics['nll']:.4f}  "
                  f"ppl={metrics['perplexity']:.4f}")
        finally:
            elapsed = time.time() - _t_start
            print(f"  Time for adapter {adapter_path}{f', ce_weight={ce_weight}' if ce_weight is not None else ''}: {format_elapsed(elapsed)}")
            free_model_memory()

    if args.use_weighted_ce:
        run_labels = ["base" if ce_w is None else str(ce_w) for _, ce_w in iter_configs]
    else:
        run_labels = [ap for ap, _ in iter_configs]
    # Paired bootstrap: same resample for all models -> CIs for each model and for differences vs base
    per_model_cis, diff_cis = paired_bootstrap_cis(per_model_metrics)
    print("\n--- Paired-bootstrap CIs (same resample per iteration; use these for comparisons) ---")
    for m, label in enumerate(run_labels):
        c = per_model_cis[m]
        print(f"  {label}: acc {c['accuracy_ci']}  ppl {c['perplexity_ci']}")
    print("--- CI of difference vs first run ---")
    for j in range(1, len(run_labels)):
        d = diff_cis[(0, j)]
        print(f"  {run_labels[j]} - {run_labels[0]}: acc {d['accuracy_ci']}  ppl {d['perplexity_ci']}")

    # Upsert long-format rows; key includes evaluation_dataset so c4 vs.
    # openwebtext eval calls for the same suite stay distinct.
    if args.csv_output:
        base_dim = ecu.base_dim_row(args, training_dataset=args.model_training_dataset)
        base_dim["evaluation_dataset"] = eval_dataset_label
        metric_cols = [
            "accuracy", "accuracy_ci_lo", "accuracy_ci_hi",
            "accuracy_diff_ci_lo", "accuracy_diff_ci_hi",
            "avg_confidence",
            "nll",
            "perplexity", "perplexity_ci_lo", "perplexity_ci_hi",
            "perplexity_diff_ci_lo", "perplexity_diff_ci_hi",
            "num_examples",
        ]
        fieldnames = ecu.COMMON_DIM_COLS + ["evaluation_dataset"] + metric_cols
        key_cols = ["training_dataset", "dataset_type", "model", "ablation",
                    "max_content_words", "evaluation_dataset"]
        new_rows = []
        for m, ((adapter_path, ce_w), metrics) in enumerate(zip(iter_configs, metrics_per_adapter)):
            c = per_model_cis[m]
            d = diff_cis.get((0, m))
            row = dict(base_dim)
            row["adapter_weight"] = adapter_path
            row["ce_weight"] = "" if ce_w is None else ce_w
            row["accuracy"] = f"{metrics['accuracy']:.6f}"
            row["accuracy_ci_lo"] = f"{c['accuracy_ci'][0]:.6f}"
            row["accuracy_ci_hi"] = f"{c['accuracy_ci'][1]:.6f}"
            row["accuracy_diff_ci_lo"] = "" if d is None else f"{d['accuracy_ci'][0]:.6f}"
            row["accuracy_diff_ci_hi"] = "" if d is None else f"{d['accuracy_ci'][1]:.6f}"
            row["avg_confidence"] = f"{metrics['avg_confidence']:.6f}"
            row["nll"] = f"{metrics['nll']:.6f}"
            row["perplexity"] = f"{metrics['perplexity']:.6f}"
            row["perplexity_ci_lo"] = f"{c['perplexity_ci'][0]:.6f}"
            row["perplexity_ci_hi"] = f"{c['perplexity_ci'][1]:.6f}"
            row["perplexity_diff_ci_lo"] = "" if d is None else f"{d['perplexity_ci'][0]:.6f}"
            row["perplexity_diff_ci_hi"] = "" if d is None else f"{d['perplexity_ci'][1]:.6f}"
            row["num_examples"] = metrics["num_examples"]
            new_rows.append(row)
        n = ecu.upsert_csv(args.csv_output, fieldnames, key_cols, base_dim, new_rows)
        print(f"Upserted {n} rows into {args.csv_output}")


if __name__ == "__main__":
    main()
